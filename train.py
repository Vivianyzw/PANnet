import torch
from data_loader import *
from loss import *
from model import *
import torch.utils.data as DATA
import os

BATCH_SIZE = 1
BASE_LR = 1e-3
G_LR = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0,2,3,4]

normalization_mean = torch.tensor([0.485, 0.456, 0.406])
normalization_std = torch.Tensor([0.229, 0.224, 0.225])


def Normalization(input, mean, std):
    m = mean.clone().detach().requires_grad_(True).view(-1, 1, 1).to(device)
    s = std.clone().detach().requires_grad_(True).view(-1, 1, 1).to(device)
    output = (input - m) / s
    return output


img_path = r'D:\work\vivian\data\ch4_training_images'
label_path = r'D:\work\vivian\data\ch4_training_localization_transcription_gt'

dataset = DataLoader(img_path, label_path, True, True)
data_loader = DATA.DataLoader(dataset, batch_size=BATCH_SIZE)
net = PANnet()
net = net.to(device)
print(net)
# gparam = list(map(id, net.features.parameters()))
# base_param = filter(lambda p: id(p) not in gparam, net.parameters())
optimizer = torch.optim.SGD(net.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=5e-4)
criterion = PANloss()

for epoch in range(1000):
    for i, data in enumerate(data_loader, 0):
        image, gt_maps, seg_maps, training_mask = data
        text_gt = gt_maps[:, :, :, 0]
        text_mask = gt_maps[:, :, :, 1]
        kernel_gt = seg_maps[:, :, :, 0]
        kernel_mask = seg_maps[:, :, :, 1]

        image = image.to(device)
        text_gt = text_gt.to(device)
        text_mask = text_mask.to(device)
        kernel_gt = kernel_gt.to(device)
        kernel_mask = kernel_mask.to(device)
        training_mask = training_mask.to(device)

        norm_image = Normalization(image, normalization_mean, normalization_std)
        text_pred, kernel_pred, similarity_vector = net(norm_image)
        loss, L_text, L_kernel, L_agg, L_dis = criterion(text_gt, text_pred, text_mask, kernel_gt, kernel_pred, kernel_mask, similarity_vector)
        print('Epoch: %d | iter: %d | train loss: %.6f | ' % (epoch, i, float(loss)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 99:
            model_name = os.path.join('model/model_%d.pkl' % epoch)
            torch.save(net.state_dict(), model_name)
