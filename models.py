import torch
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transform_patch import Block
import torch.nn as nn

from utils import take_indexes
from image_random import ImageRandom

class EncoderMAE(nn.Module):
    def __init__(self, 
                 patch_size=2,
                 embedding_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ):
        super().__init__()

        image_size = 32
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.emd_pos = nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, embedding_dim))
        self.shuffle = ImageRandom(mask_ratio)

        self.patch = nn.Conv2d(3, embedding_dim, patch_size, patch_size)

        blocks = []
        for _ in range(num_layer):
            blocks.append(Block(embedding_dim, num_head))
        self.transform_patch = nn.Sequential(*blocks)

        self.norm = nn.LayerNorm(embedding_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.emd_pos, std=.02)


    def forward(self, img):
        patches = rearrange(self.patch(img), 'b c h w -> (h w) b c') 
        patches += self.emd_pos
        patches, _, b_idx = self.shuffle(patches)
        expand_cls_token = self.cls_token.expand(-1, patches.shape[1], -1)
        patches = torch.cat([expand_cls_token, patches], dim=0)
        rg_pattern = rearrange(patches, 't b c -> b t c')
        feats = self.norm(self.transform_patch(rg_pattern))

        return rearrange(feats, 'b t c -> t b c'), b_idx

class DecoderMAE(nn.Module):
    def __init__(self, embedding_dim=192, num_layer=4,num_head=3, patch_size=2):
        super().__init__()
        image_size = 32
        self.token_masked = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        num_patch = image_size / patch_size
        nun_paras_emd_pos = torch.zeros(num_patch ** 2 + 1, 1, embedding_dim)
        self.emd_pos = nn.Parameter(nun_paras_emd_pos)

        blocks = []
        for _ in range(num_layer):
            blocks.append(Block(embedding_dim, num_head))
        self.transform_patch = nn.Sequential(*blocks)
        
        output_size = 3 * patch_size ** 2
        self.head = nn.Linear(embedding_dim, output_size)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=2, p2=2, h=16)

        trunc_normal_(self.token_masked, std=.02)
        trunc_normal_(self.emd_pos, std=.02)

        
    def forward(self, feat, b_index):
        b_index = torch.cat([torch.zeros(1, b_index.shape[1]).to(b_index), b_index + 1], dim=0)
        mask_token_expand = self.token_masked.expand(b_index.shape[0] - feat.shape[0], feat.shape[1], -1)
        feat = torch.cat([feat, mask_token_expand], dim=0)
        feat = take_indexes(feat, b_index)
        feat += self.emd_pos

        feat = self.transform_patch(rearrange(feat, 't b c -> b t c'))
        feat = rearrange(feat, 'b t c -> t b c')[1:] 

        patch = self.head(feat)
        mask = torch.zeros_like(patch)
        mask[feat.shape[0]:] = 1
        mask = take_indexes(mask, b_index[1:] - 1)
        mask = self.patch2img(mask)

        return self.patch2img(patch), mask

class MAE(nn.Module):
    def __init__(self, patch_size=2, embedding_dim=192, encoder_num_layer=12, encoder_num_head=3, decoder_num_layer=4, decoder_num_head=3, mask_ratio=0.75):
        super().__init__()
        self.encoder = EncoderMAE(patch_size, embedding_dim, encoder_num_layer, encoder_num_head, mask_ratio)
        self.decoder = DecoderMAE(patch_size, embedding_dim, decoder_num_layer, decoder_num_head)

    def forward(self, img):
        feats, b_index = self.encoder(img)
        predicted_img, mask = self.decoder(feats, b_index)
        return predicted_img, mask

class ViT_cls(nn.Module):
    def __init__(self, encoder: EncoderMAE) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.emd_pos = encoder.pos_embedding
        self.patch = encoder.patch
        self.transform_patch = encoder.transform_patch
        self.layer_norm = encoder.layer_norm
        self.head = nn.Linear(self.emd_pos.shape[-1], 10)

    def forward(self, img):
        patches = self.patch(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches += self.emd_pos
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        feats = self.layer_norm(self.transform_patch(rearrange(patches, 't b c -> b t c')))
        feats = rearrange(feats, 'b t c -> t b c')
        return self.head(feats[0])