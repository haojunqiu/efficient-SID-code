"""The augmented CLIP image/text embedder used by text-driven style transfer.

Vendored from Text2LIVE (https://github.com/omerbt/Text2LIVE, MIT license): the augmentation
templates, ``compose_text_with_templates``, the cosine loss and ``ClipExtractor`` itself, inlined
so this module is self-contained. ``patch_clip_pos_embed_interpolation`` adds DINO-style
interpolation of CLIP's positional embedding (https://github.com/facebookresearch/dino) so the
ViT accepts non-square inputs at resolutions other than 224.

This is third-party code adapted to fit, not ours to redesign. What *is* ours -- the guidance
step that consumes these embeddings -- lives in ``efficient_sid/applications/text_style.py``
beside the sampler that applies it.

Requires an openai-compatible ``clip`` (e.g. `pip install clip-anytorch`).
"""
import math
import types
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

import clip

from .timing import timed

# ---------------------------------------------------------------------------
# Text prompt templates (Text2LIVE)
# ---------------------------------------------------------------------------

_TEMPLATES_HR = [
    "photo of {}.", "high quality photo of {}.", "a photo of {}.", "the photo of {}.",
    "image of {}.", "an image of {}.", "high quality image of {}.",
    "a high quality image of {}.", "the {}.", "a {}.", "{}.", "{}", "{}!", "{}...",
]
_TEMPLATES_LR = [
    "photo of {}.", "low quality photo of {}.", "low resolution photo of {}.",
    "low-res photo of {}.", "blurry photo of {}.", "pixelated photo of {}.",
    "a photo of {}.", "the photo of {}.", "image of {}.", "an image of {}.",
    "low quality image of {}.", "a low quality image of {}.", "low resolution image of {}.",
    "a low resolution image of {}.", "low-res image of {}.", "a low-res image of {}.",
    "blurry image of {}.", "a blurry image of {}.", "pixelated image of {}.",
    "a pixelated image of {}.", "the {}.", "a {}.", "{}.", "{}", "{}!", "{}...",
]


def get_augmentations_template(flag="lr"):
    if flag == "hr":
        return _TEMPLATES_HR
    if flag == "lr":
        return _TEMPLATES_LR
    raise ValueError(f"Unknown template flag {flag!r}; expected 'hr' or 'lr'.")


def compose_text_with_templates(text, templates):
    return [t.format(text) for t in templates]


def cosine_loss(x, y, scaling=1.2):
    return scaling * (1 - F.cosine_similarity(x, y).mean())

# -------------------------------------------------------------------------
# Arbitrary-resolution CLIP: interpolate the ViT positional embedding (DINO-style)
# -------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Arbitrary-resolution CLIP: interpolate the ViT positional embedding (DINO-style)
# ---------------------------------------------------------------------------
# Stock ``clip.load`` bakes in a 224x224 positional embedding, so a non-square (or non-224)
# input mismatches the token count. Text2LIVE patches CLIP's VisionTransformer to interpolate
# the positional embedding to the actual grid (https://github.com/facebookresearch/dino). We
# monkeypatch the loaded model's ``visual.forward`` to do the same, so the augmented views
# (which are non-square) work.

def _interpolate_pos_encoding(visual, x, w, h):
    positional_embedding = visual.positional_embedding.unsqueeze(0)
    patch_size = visual.conv1.kernel_size[0]
    npatch = x.shape[1] - 1
    N = positional_embedding.shape[1] - 1
    if npatch == N and w == h:
        return positional_embedding
    class_pos_embed = positional_embedding[:, 0]
    patch_pos_embed = positional_embedding[:, 1:]
    dim = x.shape[-1]
    w0 = w // patch_size
    h0 = h // patch_size
    # add a small number to avoid floating-point error in the interpolation (see DINO issue #8)
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


def _visual_forward_interp(visual, x):
    h, w = x.shape[-2:]
    x = visual.conv1(x)                                   # [*, width, grid_h, grid_w]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [*, grid**2, width]
    x = torch.cat(
        [visual.class_embedding.to(x.dtype)
         + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
        dim=1,
    )
    x = x + _interpolate_pos_encoding(visual, x, w, h).to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)                                # NLD -> LND
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)                                # LND -> NLD
    x = visual.ln_post(x[:, 0, :])
    if visual.proj is not None:
        x = x @ visual.proj
    return x


def patch_clip_pos_embed_interpolation(model):
    """Enable arbitrary/non-square CLIP input by interpolating the ViT positional embedding."""
    model.visual.forward = types.MethodType(_visual_forward_interp, model.visual)
    return model

# ---------------------------------------------------------------------------
# ClipExtractor (Text2LIVE)
# ---------------------------------------------------------------------------

@dataclass
class ClipExtractorConfig:
    """The CLIP embedder, and the augmented views each embedding averages over."""
    model_name: str = "ViT-B/32"          # any name clip.load accepts
    n_aug: int = 16                       # augmented views averaged into one embedding
    affine_transform_fill: bool = True    # corners the augs expose: True = white, False = black


class ClipExtractor(torch.nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.model = clip.load(cfg["clip_model_name"], device=device)[0].eval().requires_grad_(False)
        patch_clip_pos_embed_interpolation(self.model)  # allow non-square / non-224 CLIP input
        self.text_criterion = cosine_loss
        self.clip_input_size = 224
        self.clip_normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        self.basic_transform = T.Compose([T.Resize(self.clip_input_size, max_size=380), self.clip_normalize])
        fill = cfg.get("clip_affine_transform_fill", True)
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.RandomAffine(degrees=15, translate=(0.1, 0.1), fill=fill,
                                          interpolation=InterpolationMode.BILINEAR)], p=0.8),
            T.RandomPerspective(distortion_scale=0.4, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=fill),
            T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.7),
            T.RandomGrayscale(p=0.15),
        ])
        self.n_aug = cfg["n_aug"]

    def augment_input(self, input, n_aug=None, clip_input_size=None):
        n_aug = self.n_aug if n_aug is None else n_aug
        clip_input_size = self.clip_input_size if clip_input_size is None else clip_input_size
        cutout = T.Resize(clip_input_size, max_size=320)(input)
        cutout_h, cutout_w = cutout.shape[-2:]
        cutouts = [self.augs(cutout)]
        sideY, sideX = input.shape[2:4]
        for _ in range(n_aug - 1):
            s = torch.zeros(1).uniform_(0.6, 1).item()
            crop = T.RandomCrop(size=(int(sideY * s), int(sideX * s)))(input)
            crop = T.Resize((cutout_h, cutout_w))(crop)
            cutouts.append(self.augs(crop))
        return torch.cat(cutouts)

    def get_image_embedding(self, x, aug=True):
        views = self.augment_input(x) if aug else self.basic_transform(x)
        return self.encode_image(self.clip_normalize(views))

    def encode_image(self, x):
        return self.model.encode_image(x)

    def get_text_embedding(self, text, template, average_embeddings=False):
        if isinstance(text, str):
            text = [text]
        embeddings = []
        for prompt in text:
            with torch.no_grad():
                emb = self.model.encode_text(
                    clip.tokenize(compose_text_with_templates(prompt, template)).to(self.device))
            embeddings.append(emb)
        embeddings = torch.cat(embeddings)
        if average_embeddings:
            embeddings = embeddings.mean(dim=0, keepdim=True)
        return embeddings

    def calculate_clip_loss(self, outputs, target_embeddings):
        n = np.random.randint(1, len(target_embeddings) + 1)
        target_embeddings = target_embeddings[torch.randint(len(target_embeddings), (n,))]
        loss = 0.0
        for img in outputs:
            img_e = self.get_image_embedding(img.unsqueeze(0))
            for target in target_embeddings:
                loss += self.text_criterion(img_e, target.unsqueeze(0))
        loss /= len(target_embeddings)
        return loss


@timed("CLIP load", load=True)
def get_clip_extractor(extractor_config, device="cuda"):
    """Our names in, Text2LIVE's dict out. ``ClipExtractor`` below is vendored and reads the
    latter, so this is the one place the two vocabularies meet."""
    return ClipExtractor(
        {"clip_model_name": extractor_config.model_name,
         "n_aug": extractor_config.n_aug,
         "clip_affine_transform_fill": extractor_config.affine_transform_fill},
        device=device,
    )
