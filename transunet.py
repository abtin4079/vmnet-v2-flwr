import torch
import torch.nn as nn
from einops import rearrange

from utils.vit import ViT


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        # self.vit_img_dim = img_dim // patch_dim
        # self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
        #                head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        # self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        # self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        # x = self.vit(x)
        # x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        # x = self.conv2(x)
        # x = self.norm2(x)
        # x = self.relu(x)

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x


class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()
        
        # for original image encoder
        self.encoder1 = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)
        # for sailency map encoder
        self.encoder2 = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        # half the shape after concatenate 
        # self.conv_layer = nn.Conv2d(in_channels=out_channels,  # Number of input channels
        #                out_channels=out_channels,  # Number of output channels
        #                kernel_size=3,  # Size of the convolving kernel
        #                stride=2,  # Stride of the convolution
        #                padding=1)  # Padding added to all four sides of the input
        

        #initialize the conv1*1 for changing the shapes of ViT input
        self.conv11 = nn.Conv2d(in_channels=out_channels * 2, out_channels= out_channels, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels=out_channels * 4, out_channels= out_channels * 2, kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels=out_channels * 8, out_channels= out_channels * 4, kernel_size=1)
        self.conv14 = nn.Conv2d(in_channels=out_channels * 16, out_channels=out_channels * 8, kernel_size=1)

        # initialize the ViT
        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        #Layers three for skip connection
        self.vit_skipcon = ViT(self.vit_img_dim * 2, out_channels * 4,out_channels * 4,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        #Layers two for skip connection
        self.vit_skipcon_2 = ViT(self.vit_img_dim * 4, out_channels * 2,out_channels * 2,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        #Layers one for skip connection
        self.vit_skipcon_1 = ViT(self.vit_img_dim * 8, out_channels, out_channels,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        # initialize the decoder
        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x, y):
        x, x1, x2, x3 = self.encoder1(x)
        y, y1, y2, y3 = self.encoder2(y)
        

        # concatenate the x's and y's to z's
        z1 = torch.cat([x1, y1], dim=1)
        # print(f'x1 shape is :{x1.shape}')
        # print(f'y1 shape is : {y1.shape}')
        # print(f'z1 shape is :{z1.shape}')

        # PASS THE FIRST SKIP CONNECTION TROUGH THE ViT
        # print(f'z1 shape is :{z1.shape}')

        #print(f'z1 shape is :{z1.shape}')
        # z1 = self.vit_skipcon_1(z1)
        # z1 = rearrange(z1, "b (x y) c -> b c x y", x=self.vit_img_dim , y=self.vit_img_dim )
        z1 = self.conv11(z1)
        #print(f'z1 shape is :{z1.shape}')

        z2 = torch.cat([x2, y2], dim=1)
        # print(f'x2 shape is :{x2.shape}')
        # print(f'y2 shape is : {y2.shape}')
        #print(f'z2 shape is :{z2.shape}')
        z2 = self.conv12(z2)
        #print(f'z2 shape is :{z2.shape}')

        # z2 = self.vit_skipcon_2(z2)
        # #print(f'z2 shape is :{z2.shape}')

        # z2 = rearrange(z2, "b (x y) c -> b c x y", x=self.vit_img_dim *4, y=self.vit_img_dim *4)
        #print(f'z2 shape is :{z2.shape}')

        z3 = torch.cat([x3, y3], dim=1)
        # print(f'x3 shape is :{x3.shape}')
        # print(f'y3 shape is : {y3.shape}')
        # print(f'z3 shape is :{z3.shape}')

        #initialize the ViT on teh lowest skip connection 
        #print(f'z3 shape is :{z3.shape}')
        z3 = self.conv13(z3)
        #print(f'z3 shape is :{z3.shape}')
        # z3 = self.vit_skipcon(z3)
        
        # z3 = rearrange(z3, "b (x y) c -> b c x y", x=self.vit_img_dim * 2, y=self.vit_img_dim * 2)
        #print(f'z3 shape is :{z3.shape}')

        # print(f'z3 shape is :{z3.shape}')


        z = torch.cat((x, y), dim=1)
        # print(f'x shape is :{x.shape}')
        # print(f'y shape is : {y.shape}')
        # print(f'z shape is :{z.shape}')
        z = self.conv14(z)
       # print(f'z shape is :{z.shape}')
        
        # pass the z to the ViT 
        z = self.vit(z)

        z = rearrange(z, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)
        #print(f'z shape is :{z.shape}')

        z = self.conv2(z)
        z = self.norm2(z)
        z = self.relu(z)



        z = self.decoder(z, z1, z2, z3)
        #print(f'z shape is :{z.shape}')
        return z


if __name__ == '__main__':
    import torch

    transunet = TransUNet(img_dim=256,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)

    #print(sum(p.numel() for p in transunet.parameters()))
    #print(transunet.forward(x= torch.rand(1, 3, 128, 128), y=torch.rand(1, 3, 128, 128)).shape)
    print(transunet(torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)).shape)
    #print(transunet)