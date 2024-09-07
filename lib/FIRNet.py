import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from lib.pvtv2 import pvt_v2_b2
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class token_TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(token_TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, fea):

        for block in self.blocks:
            fea = block(fea)

        fea = self.norm(fea)

        return fea


class token_Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_Transformer, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_s = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                                     mlp_ratio=mlp_ratio)

    def forward(self, rgb_fea):
        B, _, _ = rgb_fea.shape
        fea_1_16 = self.mlp_s(self.norm(rgb_fea))  # [B, 14*14, 384]
        fea_1_16 = self.encoderlayer(fea_1_16)
        return fea_1_16


class Transformer_Decoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads):
        super(Transformer_Decoder, self).__init__()
        self.token_trans = token_Transformer(embed_dim, depth, num_heads, mlp_ratio=3.)
        self.fc_96_384 = nn.Linear(96, 384)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.downsample4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False)
        self.pre_1_16 = nn.Linear(384, 1)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, z):
        x5 = x
        y4 = self.downsample2(y)
        z3 = self.downsample4(z)
        feat_t = torch.cat([x5, y4, z3], 1)  # [B, 96, 11, 11]
        B, Ct, Ht, Wt = feat_t.shape
        feat_t = feat_t.view(B, Ct, -1).transpose(1, 2)
        feat_t = self.fc_96_384(feat_t)  # [B, 11*11, 384]
        Tt = self.token_trans(feat_t)
        mask_x = self.pre_1_16(Tt)
        mask_x = mask_x.transpose(1, 2).reshape(B, 1, Ht, Wt)

        return mask_x




class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv1 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.conv1x1 = BasicConv2d(in_channel, out_channel, 1)
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 16, out_channel, 1, bias=False),
            nn.Sigmoid()
        )



    def forward(self, x):
        x0 = self.branch0(x)
        x4 = self.branch4(x)
        x3 = self.branch3(self.conv1x1(x) + x4)
        x2 = self.branch2(self.conv1x1(x) + x3)
        x1 = self.branch1(self.conv1x1(x) + x2)
        x5 = self.fc(self.avgpool(x))
        x_cat = self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1))
        x_cat = self.conv1(x5 + x_cat)
        x = self.relu(x_cat + x0)
        #print(x.shape)
        #print(x5.shape)
        return x




class IRM(nn.Module):
    def __init__(self, channel):
        super(IRM, self).__init__()
        self.conv1_32 = nn.Conv2d(1, 32, 1)
        self.score = nn.Conv2d(channel, 1, 3, padding=1)
        self.conv_1 = BasicConv2d(channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # global_att

        self.global_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.cbs = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.Sigmoid()
                                 )

        self.cbr_5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )



    def forward(self, x, y):
        x1 = self.conv_1(x * y)
        y1 = self.conv_2(y + x)
        xy = self.cbr_5(x1 + y1)
        xy_c = self.global_att1(xy) + x1 + y1
        out = self.cbs(xy_c)
        y = y + self.score(out)
        return y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = IRM(channel)

    def forward(self, x, y):
        y = self.weak_gra(x, y)
        return y




class Network(nn.Module):
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- Res2Net Backbone ----
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/media/lab509-1/data1/RJC/lib/pvt_v2_b2/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # ---- Receptive Field Block like module ----
        self.rfb1_1 = RFB_modified(64, channel)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)

        # ---- Transformer_Decoder ----
        self.TD = Transformer_Decoder(384, 4, 6)
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)
        self.RS2 = ReverseStage(channel)

        self.conv32_1 = nn.Conv2d(32, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Feature Extraction
        pvt = self.backbone(x)

        # Receptive Field Block
        x1_rfb = self.rfb1_1(pvt[0])  # channel -> 32
        x2_rfb = self.rfb2_1(pvt[1])  # channel -> 32
        x3_rfb = self.rfb3_1(pvt[2])  # channel -> 32
        x4_rfb = self.rfb4_1(pvt[3])  # channel -> 32



        # Transformer_Decoder
        S_g = self.TD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=32, mode='bilinear')  # Sup-1 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ----stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=1, mode='bilinear')
        ra5_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra5_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ----stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra4_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra4_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ----stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra3_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')  # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ----stage 2 ----
        guidance_3 = F.interpolate(S_3, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS2(x1_rfb, guidance_3)
        S_2 = ra3_feat + guidance_3
        S_2_pred = F.interpolate(S_2, scale_factor=4, mode='bilinear')  # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred, S_2_pred


if __name__ == '__main__':
    import numpy as np
    from time import time

    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 416, 416)
    frame_rate = np.zeros((1, 1))
    for i in range(1):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
