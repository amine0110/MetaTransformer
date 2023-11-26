import torch 
import torch.nn as nn
from timm.models.vision_transformer import Block
from Data2Seq import Data2Seq
video_tokenizer = Data2Seq(modality='video',dim=768)
# audio_tokenizer = Data2Seq(modality='audio',dim=768)
# time_series_tokenizer = Data2Seq(modality='time-series',dim=768)


video = './data/demo_arm_wrestling.mp4'
features = torch.concat([video_tokenizer(video)],dim=1) # , audio_tokenizer(audio)

#! For base-scale encoder:
ckpt = torch.load("./model_ckpt/Meta-Transformer_base_patch16_encoder.pth")
encoder = nn.Sequential(*[
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
encoder.load_state_dict(ckpt,strict=True)

#! For large-scale encoder:
'''ckpt = torch.load("Meta-Transformer_large_patch14_encoder.pth")
encoder = nn.Sequential(*[
            Block(
                dim=1024,
                num_heads=16,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(24)])
encoder.load_state_dict(ckpt,strict=True)'''

encoded_features = encoder(features)