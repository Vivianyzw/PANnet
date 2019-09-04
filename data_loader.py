import torch.utils.data as DATA
import os
import os.path
import glob
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import traceback
import pyclipper
import cv2


def make_dataset(img_path, label_path):
    dataset = []
    for img in glob.glob(os.path.join(img_path, '*.jpg')):
        basename = os.path.basename(img)
        labelname = 'gt_' + basename[:-3] + 'txt'
        image = os.path.join(img_path, basename)
        label = os.path.join(label_path, labelname)
        dataset.append([image, label])
    return dataset


class DataLoader(DATA.Dataset):
    def __init__(self, img_path, label_path, transform=None, rescale=None):
        super(mytraindata, self).__init__()
        self.train_set_path = make_dataset(img_path, label_path)
        self.transform = transform
        self.rescale = rescale
        self.scale_ratio = [1.0, 0.5]

    def __getitem__(self, item):
        img_path, label_path = self.train_set_path[item]
        image = cv2.imread(img_path)
        polys, tags = load_annotation(label_path)
        image, polys, tags = crop_area(image, polys, tags)
        input_size = image.shape[:-1]
        polys, tags = check_and_validate_polys(polys, tags, input_size)
        gt_maps, seg_maps, training_mask = generate_seg(input_size, polys, tags, self.scale_ratio)
        transform = transforms.ToTensor()
        image = cv2.resize(image, (640, 640))
        gt_maps = cv2.resize(gt_maps, (640, 640))
        seg_maps = cv2.resize(seg_maps, (640, 640))
        training_mask = cv2.resize(training_mask, (640, 640))
        
        if self.transform:
            image = transform(image)

        return image, gt_maps, seg_maps, training_mask

    def __len__(self):
        return len(self.train_set_path)


def load_annotation(p):
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys), np.array(text_tags)
    with open(p, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().strip('\ufeff').strip('\xef\xbb\xbf')
            labels = line.split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = labels[:8]
            label = ','.join(labels[8:])
            x1, y1, x2, y2, x3, y3, x4, y4 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###' or label == '?':
                text_tags.append(True)
            else:
                text_tags.append(False)
    return np.array(text_polys), np.array(text_tags)


def check_and_validate_polys(polys, tags, input_size):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = input_size
    if polys.shape[0] == 0:
        return [], []
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        if abs(pyclipper.Area(poly))<1:
            continue
        #clockwise
        if pyclipper.Orientation(poly):
            poly = poly[::-1]

        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(im, polys, tags, crop_background=False, max_tries=50, min_crop_side_ratio=0.1):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        if xmax - xmin < min_crop_side_ratio*w or ymax - ymin < min_crop_side_ratio*h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


def perimeter(poly):
    try:
        p=0
        nums = poly.shape[0]
        for i in range(nums):
            p += abs(np.linalg.norm(poly[i%nums]-poly[(i+1)%nums]))
        # logger.debug('perimeter:{}'.format(p))
        return p
    except Exception as e:
        traceback.print_exc()
        raise e


def shrink_poly(poly, r):
    try:
        area_poly = abs(pyclipper.Area(poly))
        perimeter_poly = perimeter(poly)
        poly_s = []
        pco = pyclipper.PyclipperOffset()
        if perimeter_poly:
            d=area_poly*(1-r*r)/perimeter_poly
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            poly_s = pco.Execute(-d)
        return poly_s
    except Exception as e:
        traceback.print_exc()
        raise e


#TODO:filter small text(when shrincked region shape is 0 no matter what scale ratio is)
def generate_seg(im_size, polys, tags, scale_ratio):
    '''
    :param im_size: input image size
    :param polys: input text regions
    :param tags: ignore text regions tags
    :param image_index: for log
    :param scale_ratio:ground truth scale ratio, default[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    :return:
    seg_maps: segmentation results with different scale ratio, save in different channel
    training_mask: ignore text regions
    '''
    h, w = im_size
    # mark different text poly
    # kernel层的mask，有几个scale ratio，就该生成几张
    seg_maps = np.zeros((h, w, len(scale_ratio)), dtype=np.uint8)
    gt_maps = np.zeros((h, w, len(scale_ratio)), dtype=np.uint8)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    ignore_poly_mark = []
    for i in range(len(scale_ratio)):
        seg_map = np.zeros((h, w), dtype=np.uint8)
        gt_map = np.zeros((h, w), dtype=np.uint8)
        count = 0
        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            poly = poly_tag[0]
            tag = poly_tag[1]

            # ignore ###
            if i == 0 and tag:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_poly_mark.append(poly_idx)

            # seg map
            shrinked_polys = []
            if poly_idx not in ignore_poly_mark:
                shrinked_polys = shrink_poly(poly.copy(), scale_ratio[i])

            if not len(shrinked_polys) and poly_idx not in ignore_poly_mark:
                # logger.info("before shrink poly area:{} len(shrinked_poly) is 0,image {}".format(
                #     abs(pyclipper.Area(poly)),image_name))
                # if the poly is too small, then ignore it during training
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_poly_mark.append(poly_idx)
                continue
            count += 1
            for shrinked_poly in shrinked_polys:
                seg_map = cv2.fillPoly(seg_map, [np.array(shrinked_poly).astype(np.int32)], count)
                gt_map = cv2.fillPoly(seg_map, [np.array(shrinked_poly).astype(np.int32)], 1)

        seg_maps[..., i] = seg_map
        gt_maps[..., i] = gt_map
    return gt_maps, seg_maps, training_mask


if __name__ == '__main__':
    img_path = r'D:\work\vivian\data\ch4_training_images'
    label_path = r'D:\work\vivian\data\ch4_training_localization_transcription_gt'
    # labelname = 'gt_img_1.txt'
    # scale_ratio = [1.0, 0.5]
    # image = cv2.imread(os.path.join(img_path, 'img_1.jpg'))

    # polys, tags = load_annotation(os.path.join(label_path, labelname))
    # image, polys, tags = crop_area(image, polys, tags)
    # input_size = image.shape[:-1]
    # polys, tags = check_and_validate_polys(polys, tags, input_size)
    # gt_maps, seg_maps, training_mask = generate_seg(input_size, polys, tags, 'a', scale_ratio)
    # print(seg_maps)
    # print(seg_maps.shape)
    # print(np.max(seg_maps))
    # seg_map = seg_maps[:, :, 1]*100
    # cv2.imshow('', seg_map)
    # cv2.waitKey()
    dataset = mytraindata(img_path, label_path, True, True)
    data_loader = DATA.DataLoader(dataset, batch_size=1)
    for i, data in enumerate(data_loader, 0):
        image, gt_maps, seg_maps, training_mask = data
        gt_map = gt_maps[:,:,:,0]

        print(gt_maps.shape)
        cv2.imshow('', gt_map)
        cv2.waitKey()

