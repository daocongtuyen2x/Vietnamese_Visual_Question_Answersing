# PVT.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = [
    'pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large'
]


# MLP with one hidden layer
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # dim = C in x
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Create q from x
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        # Create k,v from x, with the output dim is dim * 2 means 2 blocks of k and v.
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # sr_ratio = scale rate, compact the volume in k times, if sr_ratio = k then the input volume is reduced by k times.
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        # N = number of patches
        B, N, C = x.shape
        # Linear projection from C to dim dimensions, then reshape it into numhead and reshape to (B, num_heads, N, C//numheads) >>> Each head has its own query value
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # dim = C

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) 
            # Reduce number of patches, Output shape (B, H_reduced, W_reduced, C)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # Layer norm over the last dimension
            x_ = self.norm(x_)
            # Project to B, N, 2*dim to B, N, 2, num_heads, head_dim and reshaped to 2, B, num_heads, N, C // numheads
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # Linear layer only project the last dimension.
            # Project to B, N, 2*dim to B, N, 2, num_heads, head_dim and reshaped to 2, B, num_heads, N, C // numheads
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # @: Matrix Multiplication, only apply with the 2 last dimensions.
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, C//numheads) and (B, num_heads, N, C//numheads) >>> (B, num_heads, N, N)  <The scale ratio is equipvalent to the usage of 2D Convolution features for Patch division>
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # Dropout the attention

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, num_heads, N, N) * (B, num_heads, N, C//numheads) >>> (B, num_heads, N, C//numheads) >>> Transpose and eshape to (B, N, C)
        x = self.proj(x) # (B,N,C) >> (B,N,C)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio) # Chỉ là hidden layer dim, not the output dim, vì không truyền vào output dim nên output dim = input dim
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # (B,N,C) as output of the last operation
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # (B,N,C) as output dim

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # Chuyển integer thành tuple bậc 2
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1] # Số lượng patches trên mỗi chiều
        self.num_patches = self.H * self.W # Numpatches = H * W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) # Tạo thành embedding cho các patches luôn (Interesting!)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, embed_dim, H/patch_size, W/patch_size) >>> (B, embed_dim, num_patches) >>> (B, num_patches, embed_dim)
        # First parameter of flatten() is the started dim to be flatten, end_dim = -1 as default
        x = self.proj(x).flatten(2).transpose(1, 2)
        # Normalize for embedding dimension
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        # Reassign new dimension of the output volume

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=in_chans, # Thay patch_size ban đầu là 2
                                       embed_dim=embed_dims[0]) # Nếu như vậy patch_size đầu tiên phải là 4
        self.patch_embed2 = PatchEmbed(img_size=img_size // 2, patch_size=2, in_chans=embed_dims[0], # Thay pos_embed từ 4 8 16 thành 2 4 8
                                       embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3])

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, embed_dims[3])) # Không concatenate vào cuối nữa
        self.pos_drop4 = nn.Dropout(p=drop_rate)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule depends on depth of each block
        cur = 0
        # Block là xử lí sao khi patches embedding: N = H * W của khối patch embedding input.
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm = norm_layer(embed_dims[2])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity() # Cho num_classes < 0 để thu được khối encoding

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def _get_pos_embed(self, pos_embed, patch_embed, H, W):
    #     if H * W == self.patch_embed1.num_patches:
    #         return pos_embed
    #     else:
    #         return F.interpolate(
    #             pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
    #             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        features = []
        B = x.shape[0]
        # B = (B,224,224,3)
        # stage 1
        x, (H, W) = self.patch_embed1(x) # x.shape = (B, 3136, 64)
        features.append(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W) # Sau khi đi qua 1 block, B, N, C không thay đổi 
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (B,N,C) >>> (B,H,W,C) >>> (B, C, H, W) = (B, 64, 56, 56)
        # This is where the concept of contiguous comes in. In the example above, x is contiguous but y is not because its memory layout is different to that of a tensor of same shape made from scratch. Note that the word "contiguous" is a bit misleading because it's not that the content of the tensor is spread out around disconnected blocks of memory. Here bytes are still allocated in one block of memory but the order of the elements is different!
        # When you call contiguous(), it actually makes a copy of the tensor such that the order of its elements in memory is the same as if it had been created from scratch with the same data.

        # stage 2
        x, (H, W) = self.patch_embed2(x) # x.shape = (B, 28*28, 128)
        features.append(x)
        x = x + self.pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, (H, W) = self.patch_embed3(x) # x.shape = (B, 14 * 14, 256)
        features.append(x)
        x = x + self.pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # # stage 4
        # x, (H, W) = self.patch_embed4(x) # x.shape = (B, 7 * 7, 512)
        # # cls_tokens = self.cls_token.expand(B, -1, -1) # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        # # x = torch.cat((cls_tokens, x), dim=1) # Concat one dimension in number of patches dim
        # x = x + self.pos_embed4
        # x = self.pos_drop4(x)
        # for blk in self.block4:
        #     x = blk(x, H, W)

        # x = self.norm(x)

        return x, features

    def forward(self, x):
        x, features = self.forward_features(x)
        # x = self.head(x)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pvt_tiny(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_small(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_large(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_huge_v2(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[128, 256, 512, 768], num_heads=[2, 4, 8, 12], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 10, 60, 3], sr_ratios=[8, 4, 2, 1],
        # drop_rate=0.0, drop_path_rate=0.02)
        **kwargs)
    model.default_cfg = _cfg()

    return model
