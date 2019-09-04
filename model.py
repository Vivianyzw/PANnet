import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch


# backbone

class PANnet(nn.Module):
    def __init__(self):
        super(PANnet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upscale = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, groups=128, padding=1),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        self.downscale = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, groups=128, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def get_input(self, input):
        output = self.model.conv1(input)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        input_layer1 = self.model.layer1(output)
        input_layer2 = self.model.layer2(input_layer1)
        input_layer3 = self.model.layer3(input_layer2)
        input_layer4 = self.model.layer4(input_layer3)
        input_layer1 = self.conv1(input_layer1)
        input_layer2 = self.conv2(input_layer2)
        input_layer3 = self.conv3(input_layer3)
        input_layer4 = self.conv4(input_layer4)
        return [input_layer1, input_layer2, input_layer3, input_layer4]

    def up_scale_enhancement(self, layers):
        layer1, layer2, layer3, layer4 = layers
        upscale_layer4 = layer4

        upscale_layer3 = self.upsample(upscale_layer4)
        upscale_layer3 = torch.add(upscale_layer3, layer3)
        upscale_layer3 = self.upscale(upscale_layer3)
        upscale_layer2 = self.upsample(upscale_layer3)
        upscale_layer2 = torch.add(upscale_layer2, layer2)
        upscale_layer2 = self.upscale(upscale_layer2)

        upscale_layer1 = self.upsample(upscale_layer2)
        upscale_layer1 = torch.add(upscale_layer1, layer1)
        upscale_layer1 = self.upscale(upscale_layer1)

        return [upscale_layer1, upscale_layer2, upscale_layer3, upscale_layer4]

    def down_scale_enhancement(self, layers):
        layer1, layer2, layer3, layer4 = layers
        output_layer1 = layer1

        output_layer2 = self.upsample(layer2)
        output_layer2 = torch.add(output_layer1, output_layer2)
        output_layer2 = self.downscale(output_layer2)

        output_layer3 = self.upsample(layer3)
        output_layer3 = torch.add(output_layer2, output_layer3)
        output_layer3 = self.downscale(output_layer3)

        output_layer4 = self.upsample(layer4)
        output_layer4 = torch.add(output_layer3, output_layer4)
        output_layer4 = self.downscale(output_layer4)

        return [output_layer1, output_layer2, output_layer3, output_layer4]

    def feature_fusion_module(self, input_layers, output_layers):
        input_layer1, input_layer2, input_layer3, input_layer4 = input_layers
        output_layer1, output_layer2, output_layer3, output_layer4 = output_layers
        layer1 = torch.add(input_layer1, output_layer1)
        layer2 = torch.add(input_layer2, output_layer2)
        layer3 = torch.add(input_layer3, output_layer3)
        layer4 = torch.add(input_layer4, output_layer4)

        layer2 = self.upsample(layer2)
        layer3 = self.upsample1(layer3)
        layer4 = self.upsample2(layer4)

        final_feature_map = torch.cat([layer1, layer2, layer3, layer4], 1)
        return final_feature_map

    def output_text_region(self, feature):
        return self.output(feature)
        # return text_region

    def output_kernel(self, feature):
        return self.output(feature)

    def output_similarity_vector(self, feature):
        return self.output(feature)

    def forward(self, input):
        input_layers = self.get_input(input)
        layers = self.up_scale_enhancement(input_layers)
        output_layers = self.down_scale_enhancement(layers)
        final_feature_map = self.feature_fusion_module(input_layers, output_layers)
        final_feature_map = self.upsample1(final_feature_map)
        output_text_region = self.output_text_region(final_feature_map)
        output_kernel = self.output_kernel(final_feature_map)
        output_similarity_vector = self.output_similarity_vector(final_feature_map)
        # return input_layers
        return output_text_region, output_kernel, output_similarity_vector


if __name__ == '__main__':
    import cv2
    from torchsummary import summary

    model = PANnet()
    input = torch.randn(size=(1, 3, 224, 224), dtype=torch.float32)
    # print(model)
    summary(model, (3, 224, 224),  device='cpu')
