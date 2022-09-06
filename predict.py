"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""


import glob
import os
import sys
import paddle
import paddle.nn as nn
import numpy as np
import cv2
from paddle.nn import functional as F

##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Layer):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_du = nn.Sequential(nn.Conv2D(in_channels, d, 1, padding=0, bias_attr=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.LayerList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2D(d, in_channels, kernel_size=1, stride=1,bias_attr=bias))
        
        self.softmax = nn.Softmax(axis=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = paddle.concat(inp_feats, axis=1)
        inp_feats = inp_feats.reshape([batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3]])
        
        feats_U = paddle.sum(inp_feats, axis=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = paddle.concat(attention_vectors, axis=1)
        attention_vectors = attention_vectors.reshape([batch_size, self.height, n_feats, 1, 1])
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = paddle.sum(inp_feats*attention_vectors, axis=1)
        
        return feats_V

class ContextBlock(nn.Layer):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2D(n_feat, 1, kernel_size=1, bias_attr=bias)
        self.softmax = nn.Softmax(axis=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2D(n_feat, n_feat, kernel_size=1, bias_attr=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2D(n_feat, n_feat, kernel_size=1, bias_attr=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.shape
        input_x = x
        # [N, C, H * W]
        input_x = input_x.reshape([batch, channel, height * width])
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.reshape([batch, 1, height * width])
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = paddle.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.reshape([batch, channel, 1, 1])

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x


##########################################################################
##---------- Spatial Attention ----------
class RCB(nn.Layer):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2D(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias_attr=bias, groups=groups),
            act,
            nn.Conv2D(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias_attr=bias, groups=groups)
        )

        self.act = act

        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res


##########################################################################
##---------- Resizing Layers ----------    
class Down(nn.Layer):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2D(2, ceil_mode=True, exclusive=False),
            nn.Conv2D(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias_attr=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Layer):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        Layers_body = []
        for i in range(self.scale_factor):
            Layers_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*Layers_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Up(nn.Layer):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2D(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias_attr=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Layer):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        Layers_body = []
        for i in range(self.scale_factor):
            Layers_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*Layers_body)

    def forward(self, x):
        x = self.body(x)
        return x


##########################################################################
class GRU_sample(nn.Layer):
    def __init__(self):
        super(GRU_sample, self).__init__()
    def forward(self,x,h_t_1):
        C = x.shape[1]
        z_t = F.sigmoid(x + h_t_1)
        h_hat_t = F.tanh(x + paddle.matmul(z_t, h_t_1))
        h_t = paddle.matmul((1 - z_t), h_t_1) + paddle.matmul(z_t, h_hat_t)
        #conv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=1, stride=1) 
        conv = nn.Conv2D(C, C, 1, stride=1, padding=0, bias_attr=False)
        y = conv(h_t)
        return y, h_t 
##---------- Multi-Scale Resiudal Block (MRB) ----------
class MRB(nn.Layer):
    def __init__(self, n_feat, height, width, chan_factor, bias, groups):
        super(MRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width

        self.dau_top = RCB(int(n_feat * chan_factor ** 0), bias=bias, groups=groups)
        self.dau_mid = RCB(int(n_feat * chan_factor ** 1), bias=bias, groups=groups)
        self.dau_bot = RCB(int(n_feat * chan_factor ** 2), bias=bias, groups=groups)

        self.down2 = DownSample(int((chan_factor ** 0) * n_feat), 2, chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor ** 0) * n_feat), 2, chan_factor),
            DownSample(int((chan_factor ** 1) * n_feat), 2, chan_factor)
        )

        self.up21_1 = UpSample(int((chan_factor ** 1) * n_feat), 2, chan_factor)
        self.up21_2 = UpSample(int((chan_factor ** 1) * n_feat), 2, chan_factor)
        self.up32_1 = UpSample(int((chan_factor ** 2) * n_feat), 2, chan_factor)
        self.up32_2 = UpSample(int((chan_factor ** 2) * n_feat), 2, chan_factor)

        self.conv_out = nn.Conv2D(n_feat, n_feat, kernel_size=1, padding=0, bias_attr=bias)

        # only two inputs for SKFF
        self.skff_top = SKFF(int(n_feat * chan_factor ** 0), 2)
        self.skff_mid = SKFF(int(n_feat * chan_factor ** 1), 2)
        self.convGru = GRU_sample()

    def forward(self, x):
        x_top = x.clone()
        x_mid = self.down2(x_top)
        x_bot = self.down4(x_top)

        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.skff_mid([x_mid, self.up32_1(x_bot)])
        x_top = self.skff_top([x_top, self.up21_1(x_mid)])


        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.skff_mid([x_mid, self.up32_2(x_bot)])
        #x_m,x_up = self.convGru(x_mid, self.up32_2(x_bot)) # add
        #x_mid = self.skff_mid([x_m, x_up])
        x_top = self.skff_top([x_top, self.up21_2(x_mid)])
        

        out = self.conv_out(x_top)
        out = out + x

        return out


##########################################################################
##---------- Recursive Residual Group (RRG) ----------
class RRG(nn.Layer):
    def __init__(self, n_feat, n_MRB, height, width, chan_factor, bias=False, groups=1):
        super(RRG, self).__init__()

        Layers_body = [MRB(n_feat, height, width, chan_factor, bias, groups) for _ in range(n_MRB)]
        Layers_body.append(nn.Conv2D(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias_attr=bias))
        self.body = nn.Sequential(*Layers_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
##---------- MIRNet_V2  -----------------------
class MIRNet_v2(nn.Layer):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 n_feat=80,
                 chan_factor=1.5,
                 n_MRB=2,
                 height=3,
                 width=2,
                 bias=False,
                 task=None
                 ):
        super(MIRNet_v2, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(fan_in=10000.), nn.initializer.Constant(0.0))

        self.task = task
        self.conv_in = nn.Conv2D(inp_channels, n_feat, kernel_size=3, padding=1, bias_attr=bias)

        layers_body = []

        layers_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=1))
        layers_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=2))
        layers_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))
        # layers_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))

        self.body = nn.Sequential(*layers_body)

        self.conv_out = nn.Conv2D(n_feat, out_channels, kernel_size=3, padding=1, bias_attr=bias)

    def forward(self, inp_img):
        shallow_feats = self.conv_in(inp_img)
        deep_feats = self.body(shallow_feats)

        if self.task == 'defocus_deblurring':
            deep_feats += shallow_feats
            out_img = self.conv_out(deep_feats)

        else:
            out_img = self.conv_out(deep_feats)
            out_img += inp_img

        return out_img

def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    
    #x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.reshape([B, C, H // window_size, window_size, W // window_size, window_size])
    #windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    windows = x.transpose([0, 2, 4, 1, 3, 5]).reshape([-1, C, window_size, window_size])
    return windows
    
def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return paddle.concat([x_main, x_r, x_d, x_dd], axis=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return paddle.concat([x_main, x_r], axis=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return paddle.concat([x_main, x_d], axis=0), [b_main, b_d]

def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    C = windows.shape[1]
    # print('C: ', C)

    #x = windows.view(-1, H // window_size, W // window_size, C, window_size, window_size)
    x = windows.reshape([-1, H // window_size, W // window_size, C, window_size, window_size])
    #x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    x = x.transpose([0, 3, 1, 4, 2, 5]).reshape([-1, C, H, W])
    return x

def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    res = paddle.zeros([B, C, H, W])
    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res

import time
def process(src_image_dir, save_dir):
    model = model = MIRNet_v2(n_feat=64)
    param_dict = paddle.load('./MIRnetV2_model_5.pdparams')
    model.load_dict(param_dict)
    model.eval()
    image_paths = glob.glob(os.path.join(src_image_dir, "*.png"))
    for image_path in image_paths:
        
        # img = cv2.imread(image_path)
        # h, w, c = img.shape
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = paddle.vision.transforms.resize(img, (512,512), interpolation='bilinear')
        # img = img.transpose((2,0,1))
        # img = img/255

        # img = paddle.to_tensor(img).astype('float32')
        # img = img.reshape([1]+img.shape)
        # pre = model(img)[0].numpy()

        # pre = pre.squeeze()
        # pre[pre>0.9]=1
        # pre[pre<0.1]=0
        # pre = pre*255.
        # pre = pre.transpose((1,2,0))
        # pre = paddle.vision.transforms.resize(pre, (h,w), interpolation='bilinear')
        # out_image = cv2.cvtColor(pre, cv2.COLOR_RGB2BGR)


        img = cv2.imread(image_path)
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = paddle.vision.transforms.resize(img, (2048,1024), interpolation='bilinear')
        img = img.transpose((2,0,1))
        img = img/255

        img = paddle.to_tensor(img).astype('float32')
        img = img.reshape([1]+img.shape)
        _, _, Hx, Wx = img.shape
        input_re, batch_list = window_partitionx(img, 512)
        #print(input_re.shape[0])
        for i in range(4):
            #x= input_re[2*i:2*i+1,:,:,:].unsqueeze(axis=0)
            x= input_re[2*i:2*i+1,:,:,:]
            input_re[2*i:2*i+1,:,:,:] = model(x)[0].numpy()
        pre = window_reversex(input_re, 512, Hx, Wx, batch_list)
        pre = pre.squeeze()
        pre[pre>0.95]=1
        pre[pre<0.05]=0
        pre = pre*255.
        pre = paddle.vision.transforms.resize(pre, (h,w), interpolation='bilinear')  
        pre = pre.transpose((1,2,0))     
        pre = np.float32(pre)
        out_image = cv2.cvtColor(pre, cv2.COLOR_RGB2BGR)

        # 保存结果图片
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, out_image)
        

if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    process(src_image_dir, save_dir)


