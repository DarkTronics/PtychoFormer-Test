import torch
from einops import rearrange
from torch import nn
from typing import List
from typing import Iterable
import math
from torchvision.ops import StochasticDepth

def init_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
          nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.MultiheadAttention):
      nn.init.xavier_uniform_(m.out_proj.weight)
      if m.out_proj.bias is not None:
          nn.init.constant_(m.out_proj.bias, 0)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
class OverlapPatchMerging(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=overlap_size, padding=patch_size // 2, bias=False),
            LayerNorm2d(out_channels)
        )

class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio),
            LayerNorm2d(channels)
            )
        self.att = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out
    
class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels * expansion, kernel_size=3, groups=channels, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, kernel_size=1)
        )

class ResidualAdd(nn.Module):
    """Just an util layer"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

class PtychoFormerEncoderBlock(nn.Sequential):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8, mlp_expansion: int = 4, drop_path_prob: float = .0):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )

class PtychoFormerEncoderStage(nn.Sequential):
    def __init__(self,in_channels: int, out_channels: int, patch_size: int, overlap_size: int,
        drop_probs: List[int], depth: int = 2, reduction_ratio: int = 1, num_heads: int = 8, mlp_expansion: int = 4):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(in_channels, out_channels, patch_size, overlap_size)
        self.blocks = nn.Sequential(
            *[
                PtychoFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)

def chunks(data: Iterable, sizes: List[int]):
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk
        
class PtychoFormerEncoder(nn.Module):
    def __init__(self, in_channels: int, widths: List[int], depths: List[int], all_num_heads: List[int], patch_sizes: List[int], 
                 overlap_sizes: List[int], reduction_ratios: List[int], mlp_expansions: List[int], drop_prob: float = .0):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs =  [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                PtychoFormerEncoderStage(*args)
                for args in zip(
                    [in_channels, *widths], widths, patch_sizes, overlap_sizes, chunks(drop_probs, sizes=depths),
                    depths, reduction_ratios, all_num_heads, mlp_expansions
                )
            ]
        )
        
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=1),)

class Upsample(nn.Module):
    def __init__(self, out_channels: int, widths: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [UpsampleBlock(in_channels, out_channels)
                for in_channels in widths])
    
    def forward(self, features):
        new_features = []
        out_size = features[-1].shape[-2:]
        for feature, stage in zip(features,self.stages):
            x = nn.functional.interpolate(feature, size=out_size, mode='bicubic')
            x = stage(x)
            new_features.append(x)

        return new_features
    
class Decoder(nn.Module):
    def __init__(self, channels: int, num_features: int = 4):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channels//2),
            nn.GELU(),
            nn.Conv2d(channels//2, 2, kernel_size=1, bias=False)
        )
    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.conv_block(x)
        return x

class PtychoFormer(nn.Module):
    def __init__(self, 
                 in_channels: int = 25, widths: List[int] = [64, 128, 256, 512], 
                 depths: List[int] = [3, 4, 6, 3], all_num_heads: List[int] = [1, 2, 4, 8],
                 patch_sizes: List[int] = [5, 3, 3, 3], overlap_sizes: List[int] = [2, 2, 2, 2],
                 reduction_ratios: List[int] = [8, 4, 2, 1], mlp_expansions: List[int] = [4, 4, 4, 4],
                 decoder_channels: int = 256, drop_prob: float = 0.1):
                #  in_channels: int = 9, # input channel number
                #  widths: List[int] = [32, 64, 128],  # width of model at diff stages, from [64, 128, 256, 512]
                #  depths: List[int] = [2, 3, 4],      # depth of model, from [3, 4, 6, 3]
                #  all_num_heads: List[int] = [1, 2, 4], # num of attention head in transformer layer, from [1, 2, 4, 8]
                #  patch_sizes: List[int] = [5, 3, 3], # bigger patch initially to smaller, from [7, 5, 3]
                #  overlap_sizes: List[int] = [2, 2, 2], #define stride with which patches are extracted, from [4, 2, 2]
                #  reduction_ratios: List[int] = [8, 4, 2], #define how much channels are reduced, from [8, 4, 2]
                #  mlp_expansions: List[int] = [2, 2, 2], #from, [4, 4, 4, 4]
                #  decoder_channels: int = 128,        # num channel in decoder, upsample and extract encoder, from 256
                #  drop_prob: float = 0.1): # regularization technique, drop path probability
        super().__init__()
        self.encoder = PtychoFormerEncoder(in_channels, widths, depths, all_num_heads,patch_sizes,
            overlap_sizes, reduction_ratios, mlp_expansions, drop_prob)
        self.decoder = Upsample(decoder_channels, widths[::-1])
        self.head = Decoder(decoder_channels, num_features=len(widths))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.pi = torch.tensor(math.pi)
        self.apply(init_weights)

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        x = self.head(features)
        x_updated = torch.empty_like(x)
        x_updated[:,0] = self.pi * self.tanh(x[:,0]) # phase: -pi to pi
        x_updated[:,1] = self.sigmoid(x[:,1]) # amp: 0 to 1
        return x_updated