import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class PANloss(nn.Module):

    def __init__(self):
        super(PANloss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def text_region_loss(self, gt, pred):
        text_region_loss = 1 - 2 * gt.mul(pred).sum() / (gt.mul(gt).sum() + pred.mul(pred).sum())
        return text_region_loss

    def kernel_loss(self, gt, pred):
        kernel_loss = 1 - 2 * gt.mul(pred).sum() / (gt.mul(gt).sum() + pred.mul(pred).sum())
        return kernel_loss

    # 先实现batch_size = 1损失函数
    def aggregation_loss(self, text_mask_map, kernel_mask_map, similarity_vector, batch_size=1):
        # similarity_vecor (w x h x 4)
        # 计算agg_loss时只考虑文字部分
        similarity_vector = similarity_vector.mul(text_mask_map.float())
        delta_agg = 0.5
        max_id = torch.max(text_mask_map)

        G_value = []
        loss_agg = 0.0
        for mask_id in range(1, int(max_id+1)):
            text_mask = text_mask_map == mask_id
            kernel_mask = kernel_mask_map == mask_id

            F_kernel = similarity_vector.mul(kernel_mask.float())

            G = F_kernel / kernel_mask.sum()

            G = G.reshape(batch_size, 4, -1)
            G = G.sum(dim=2)
            G_value.append(G)

            F_text = similarity_vector.mul(text_mask.float()).reshape(batch_size, 4, -1)
            G = G.unsqueeze(2)

            D = torch.norm(F_text-G, p=2, dim=1)
            D = D-delta_agg
            D = torch.where(D < 0, torch.full_like(D, 0), D)
            D = D.mul(D)

            D = D.reshape(batch_size, -1)
            l_agg = torch.log(D+1).sum(dim=1) / text_mask.sum()
            loss_agg += l_agg

        # # 这里需要注意一下，如果没有文本框, 除以max_id就为0，nan
        return loss_agg / (max_id+1e-8), G_value

    def distance_loss(self, G_value):
        delta_dis = 3
        N = len(G_value)
        L_dis = 0.0
        for i in range(N):
            for j in range(i+1, N):
                D = delta_dis - torch.norm(G_value[i]-G_value[j], dim=1)
                D = torch.where(D < 0, torch.full_like(D, 0), D)
                D = D.mul(D)
                L_dis += D
        return L_dis

    def forward(self, text_gt, text_pred, text_mask, kernel_gt, kernel_pred, kernel_mask, similarity_vector):
        """
        :param text_region_gt: 文本区域的ground truth，0：非文本；1：文本
        :param text_region: 文本区域的pred
        :param text_mask_map: 文本区域的mask，非文本：0；第一个区域：1；第二个区域:2...
        :param kernel_gt: 缩小了scale_ratio的ground truth 0：非文本；1：文本
        :param kernel_map: 预测的
        :param kernel_mask_map:
        :param similarity_vector:
        :return:
        """
        alpha = 0.5
        beta = 0.25
        L_text = self.text_region_loss(text_gt.float(), text_pred)
        L_kernel = self.kernel_loss(kernel_gt.float(), kernel_pred)
        L_agg, G_value = self.aggregation_loss(text_mask,  kernel_mask, similarity_vector)
        L_dis = self.distance_loss(G_value)
        loss = L_text + alpha * L_kernel + beta * (L_agg + L_dis)
        return loss, L_text, L_kernel, L_agg, L_dis


if __name__ == '__main__':
    import cv2
    from torchsummary import summary

    model = PANloss()

    text_region = torch.rand(size=(1, 1, 4, 4), dtype=torch.float32, requires_grad=True)
    text_gt = torch.rand(size=(1, 1, 4, 4), dtype=torch.float32, requires_grad=True)
    text_mask_map = torch.zeros(size=(1, 1, 4, 4), dtype=torch.float32, requires_grad=False)
    text_mask_map[0, 0, :, 0] = 2
    text_mask_map[0, 0, :, 1:3] = 1
    print(text_mask_map)
    kernel_map = torch.rand(size=(1, 1, 4, 4), dtype=torch.float32, requires_grad=True)
    kernel_gt = torch.rand(size=(1, 1, 4, 4), dtype=torch.float32, requires_grad=True)
    kernel_mask_map = torch.zeros(size=(1, 1, 4, 4), dtype=torch.float32, requires_grad=False)
    kernel_mask_map[0, 0, :, 0] = 2
    kernel_mask_map[0, 0, :, 1] = 1
    print(kernel_mask_map)
    similarity_vector = torch.rand(size=(1, 4, 4, 4), dtype=torch.float32, requires_grad=True)
    similarity_vector = (similarity_vector*10).int().float()
    loss = model(text_gt, text_region, text_mask_map, kernel_gt, kernel_map, kernel_mask_map, similarity_vector)
    print(loss)
    loss.backward()
    # summary(model, [(3, 224, 224), (3, 224, 224)],  device='cpu')
