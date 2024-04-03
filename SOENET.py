import torch
from fastai.vision import *
from torch import nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        nb_filter2 = [16]
        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.VGGconv_3cto32c = VGGBlock(input_channels, 6, 6)

        self.VGGconv_512cto512c = VGGBlock(18, 18, 18)

        self.VGGconv_256cto256c = VGGBlock(54, 54, 54)

        self.VGGconv_128cto128c = VGGBlock(162, 162, 162)

        self.VGGconv_64cto64c = VGGBlock(486, 486, 486)

        self.VGGconv_1944cto486c = VGGBlock(1944, 486, 486)

        self.VGGconv_648cto162c = VGGBlock(648, 162, 162)

        self.VGGconv_216cto54c = VGGBlock(216, 54, 54)

        self.VGGconv_72cto18c = VGGBlock(72, 18, 18)

        self.last = nn.Conv2d(18, num_classes, kernel_size=1)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(18, num_classes, kernel_size=1)


    def forward(self, image512,image1024,image256):
        # up-特征图长宽x2，通道不变
        # pool-特征图长宽/2，通道不变

        # print('image1024:', image1024.shape)# 1 3 1024 1024
        #-------------------------------------------------------------------分界线----------------------------------------
        L1_f1024c3_to_f512_c3 = self.pool(image1024)  # 1 32 512 512
        print('L1_f1024c3_to_f512_c3:', L1_f1024c3_to_f512_c3.shape)

        # L2不需要转512

        L3_f256c3_to_f512_c3 = self.up(image256)
        print('L3_f256c3_to_f512_c3:', L3_f256c3_to_f512_c3.shape)
        #-------------------------------------------------------------------分界线----------------------------------------
        L1_f512c3_to_f512_c6 = self.VGGconv_3cto32c(L1_f1024c3_to_f512_c3)
        print('L1_f512c3_to_f512_c6:', L1_f512c3_to_f512_c6.shape)

        L2_f512c3_to_f512_c6 = self.VGGconv_3cto32c(image512)
        print('L2_f512c3_to_f512_c6:', L2_f512c3_to_f512_c6.shape)

        L3_f512c3_to_f512_c6 = self.VGGconv_3cto32c(L3_f256c3_to_f512_c3)
        print('L3_f512c3_to_f512_c6:', L3_f512c3_to_f512_c6.shape)

        CAT1_f512c18 = torch.cat([L1_f512c3_to_f512_c6, L2_f512c3_to_f512_c6, L3_f512c3_to_f512_c6], 1)
        print('CAT1_f512c18:', CAT1_f512c18.shape)
        #-------------------------------------------------------------------分界线----------------------------------------


        L1_f512c18_to_f256_c18 = self.pool(CAT1_f512c18)
        print('L1_f512c18_to_f256_c18:', L1_f512c18_to_f256_c18.shape)


        L2_f512c18_to_f256_c18 = self.pool(self.VGGconv_512cto512c(CAT1_f512c18))
        print('L2_f512c18_to_f256_c18:', L2_f512c18_to_f256_c18.shape)

        L3_f512c18_to_f128_c18 = self.pool(self.pool(self.VGGconv_512cto512c(CAT1_f512c18)))
        print('L3_f512c18_to_f128_c18:', L3_f512c18_to_f128_c18.shape)

        #-------------------------------------------------------------------分界线----------------------------------------

        L3_f128c18_to_f256_c18 = self.up(L3_f512c18_to_f128_c18)
        print('L3_f128c18_to_f256_c18:', L3_f128c18_to_f256_c18.shape)

        CAT2_f256c54 = torch.cat([L1_f512c18_to_f256_c18, L2_f512c18_to_f256_c18, L3_f128c18_to_f256_c18], 1)
        print('CAT2_f256c54:', CAT2_f256c54.shape)
        #-------------------------------------------------------------------分界线----------------------------------------


        L1_f256c54_to_f128_c54 = self.pool(CAT2_f256c54)
        print('L1_f256c54_to_f128_c54:', L1_f256c54_to_f128_c54.shape)

        L2_f256c54_to_f128_c54 = self.pool(self.VGGconv_256cto256c(CAT2_f256c54))
        print('L2_f256c54_to_f128_c54:', L2_f256c54_to_f128_c54.shape)

        L3_f256c54_to_f64_c54 = self.pool(self.pool(self.VGGconv_256cto256c(CAT2_f256c54)))
        print('L3_f256c54_to_f64_c54:', L3_f256c54_to_f64_c54.shape)

        #-------------------------------------------------------------------分界线----------------------------------------

        L3_f64c54_to_f128_c54 = self.up(L3_f256c54_to_f64_c54)
        print('L3_f64c54_to_f128_c54:', L3_f64c54_to_f128_c54.shape)

        CAT3_f128c162 = torch.cat([L1_f256c54_to_f128_c54, L2_f256c54_to_f128_c54, L3_f64c54_to_f128_c54], 1)
        print('CAT3_f128c162:', CAT3_f128c162.shape)

        #-------------------------------------------------------------------分界线----------------------------------------

        L1_f128c162_to_f64_c216 = self.pool(CAT3_f128c162)
        print('L1_f128c162_to_f64_c216:', L1_f128c162_to_f64_c216.shape)

        L2_f128c162_to_f64_c216 = self.pool(self.VGGconv_128cto128c(CAT3_f128c162))
        print('L2_f128c162_to_f64_c216:', L2_f128c162_to_f64_c216.shape)

        L3_f128c162_to_f32_c216 = self.pool(self.pool(self.VGGconv_128cto128c(CAT3_f128c162)))
        print('L3_f128c162_to_f32_c216:', L3_f128c162_to_f32_c216.shape)

        #-------------------------------------------------------------------分界线----------------------------------------

        L3_f32c216_to_f64_c216 = self.up(L3_f128c162_to_f32_c216)
        print('L3_f32c216_to_f64_c216:', L3_f32c216_to_f64_c216.shape)

        CAT4_f64c486 = torch.cat([L1_f128c162_to_f64_c216, L2_f128c162_to_f64_c216, L3_f32c216_to_f64_c216], 1)
        print('CAT4_f64c486:', CAT4_f64c486.shape)

        #-------------------------------------------------------------------分界线--------------------------------------------------------------------------

        L1_f64c486_to_f32_c486 = self.pool(CAT4_f64c486)
        print('L1_f64c486_to_f32_c486:', L1_f64c486_to_f32_c486.shape)

        L2_f64c486_to_f32_c486 = self.pool(self.VGGconv_64cto64c(CAT4_f64c486))
        print('L2_f64c486_to_f32_c486:', L2_f64c486_to_f32_c486.shape)

        L3_f64c486_to_f16_c486 = self.pool(self.pool(CAT4_f64c486))
        print('L3_f64c486_to_f16_c486:', L3_f64c486_to_f16_c486.shape)

        #-------------------------------------------------------------------分界线----------------------------------------

        L3_f16c486_to_f32_c486 = self.up(L3_f64c486_to_f16_c486)
        print('L3_f16c486_to_f32_c486:', L3_f16c486_to_f32_c486.shape)

        CAT5_f32c1458 = torch.cat([L1_f64c486_to_f32_c486, L2_f64c486_to_f32_c486, L3_f16c486_to_f32_c486], 1)
        print('CAT5_f32c1458:', CAT5_f32c1458.shape)

        #-------------------------------------------------------------------分界线----------------------------------------

        f64c1458 = self.up(CAT5_f32c1458)
        print('f64c1458:', f64c1458.shape)

        CAT5_CAT4_f64c1944 = torch.cat([f64c1458, CAT4_f64c486], 1)
        print('CAT5_CAT4_f64c1944:', CAT5_CAT4_f64c1944.shape)

        CAT5_CAT4_f64c486 = self.VGGconv_1944cto486c(CAT5_CAT4_f64c1944)
        print('CAT5_CAT4_f64c486:', CAT5_CAT4_f64c486.shape)
        #-------------------------------------------------------------------分界线----------------------------------------
        CAT5_CAT4_f64c486tof128c486 = self.up(CAT5_CAT4_f64c486)
        print('CAT5_CAT4_f64c486tof128c486:', CAT5_CAT4_f64c486tof128c486.shape)

        CAT4_CAT3_f128c648 = torch.cat([CAT3_f128c162, CAT5_CAT4_f64c486tof128c486], 1)
        print('CAT4_CAT3_f128c648:', CAT4_CAT3_f128c648.shape)

        CAT4_CAT3_f128c162 = self.VGGconv_648cto162c(CAT4_CAT3_f128c648)
        print('CAT4_CAT3_f128c162:', CAT4_CAT3_f128c162.shape)

        #-------------------------------------------------------------------分界线----------------------------------------
        CAT4_CAT3_f128c162tof256c162 = self.up(CAT4_CAT3_f128c162)
        print('CAT4_CAT3_f128c162tof256c162:', CAT4_CAT3_f128c162tof256c162.shape)

        CAT3_CAT2_f256c216 = torch.cat([CAT2_f256c54, CAT4_CAT3_f128c162tof256c162], 1)
        print('CAT3_CAT2_f256c216:', CAT3_CAT2_f256c216.shape)

        CAT3_CAT2_f256c54 = self.VGGconv_216cto54c(CAT3_CAT2_f256c216)
        print('CAT3_CAT2_f256c54:', CAT3_CAT2_f256c54.shape)
        #-------------------------------------------------------------------分界线----------------------------------------
        CAT3_CAT2_f256c54tof512c54 = self.up(CAT3_CAT2_f256c54)
        print('CAT3_CAT2_f256c54tof512c54:', CAT3_CAT2_f256c54tof512c54.shape)

        CAT2_CAT1_f512c72 = torch.cat([CAT1_f512c18, CAT3_CAT2_f256c54tof512c54], 1)
        print('CAT2_CAT1_f512c72:', CAT2_CAT1_f512c72.shape)

        CAT2_CAT1_f512c18 = self.VGGconv_72cto18c(CAT2_CAT1_f512c72)
        print('CAT2_CAT1_f512c18:', CAT2_CAT1_f512c18.shape)

        OUTPUT = self.last(CAT2_CAT1_f512c18)

        return OUTPUT




