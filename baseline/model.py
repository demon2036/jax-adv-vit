import copy

import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

# 这里可以用两个timm模型进行构建我们的结果
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from functools import partial


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)  # 打乱index
    backward_indexes = np.argsort(forward_indexes)  # 得到原来index的位置，方便进行还原
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape  # length, batch, dim

        # B,N,D = patches.shape  # length, batch, dim

        remain_T = int(T * (1 - self.ratio))

        noise = torch.rand(T, B, device=patches.device)
        forward_indexes = torch.argsort(noise, dim=0)
        backward_indexes = torch.argsort(forward_indexes, dim=0)

        patches = take_indexes(patches, forward_indexes)[:remain_T]
        # patches=torch.gather(patches,dim=1,index=ids_shuffle)

        """
        print(patches.shape, ids_shuffle.shape, ids_restore.shape)


        while True:
            pass

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)

        patches = take_indexes(patches, forward_indexes)  # 随机打乱了数据的patch，这样所有的patch都被打乱了
        patches = patches[:remain_T]  # 得到未mask的pacth [T*0.25, B, C]
        """

        return patches, forward_indexes, backward_indexes


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))

        # 对patch进行shuffle 和 mask
        self.shuffle = PatchShuffle(mask_ratio)

        # 这里得到一个 (3, dim, patch, patch)
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        # ViT的laynorm
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    # 初始化类别编码和向量编码
    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size,
                                   h=image_size // patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat(
            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat(
            [features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)],
            dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding  # 加上了位置编码的信息

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove global feature 去掉全局信息，得到图像信息

        patches = self.head(features)  # 用head得到patchs
        mask = torch.zeros_like(patches)
        mask[T:] = 1  # mask其他的像素全部设为 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)  # 得到 重构之后的 img
        mask = self.patch2img(mask)

        return img, mask


class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask


class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.distill_token = copy.deepcopy(encoder.cls_token)
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    # def forward(self, img):
    #     patches = self.patchify(img)
    #     # patches = rearrange(patches, 'b c h w -> (h w) b c')
    #     patches = rearrange(patches, 'b c h w ->  b (h w) c')
    #
    #     patches = patches + self.pos_embedding
    #
    #     patches = rearrange(patches, 'b n c -> n b c')
    #
    #     patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
    #     patches = rearrange(patches, 't b c -> b t c')
    #     features = self.layer_norm(self.transformer(patches))
    #     features = rearrange(features, 'b t c -> t b c')
    #     logits = self.head(features[0])
    #     return logits



    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        # patches = rearrange(patches, 'b c h w ->  b (h w) c')
        patches = patches + self.pos_embedding
        # patches = rearrange(patches, 'b n c -> n b c')
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1),self.distill_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        logits = self.head(features[0])
        return logits
        # logits_distill = self.head(features[1])
        # if self.training:
        #
        #     return logits,logits_distill
        # else:
        #     return logits#,(logits_distill+logits)/2


MAE_ViT_2_baby = partial(MAE_ViT, emb_dim=192, encoder_head=3, decoder_head=3, encoder_layer=1)

MAE_ViT_2_T = partial(MAE_ViT, emb_dim=192, encoder_head=3, decoder_head=3, decoder_layer=2)
MAE_ViT_2_S = partial(MAE_ViT, emb_dim=384, encoder_head=6, decoder_head=6, decoder_layer=2)
MAE_ViT_2_M = partial(MAE_ViT, emb_dim=512, encoder_head=8, decoder_head=8, decoder_layer=2)
MAE_ViT_2_B = partial(MAE_ViT, emb_dim=768, encoder_head=12, decoder_head=12, decoder_layer=2)

if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
