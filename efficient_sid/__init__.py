"""Efficient Single-Image Diffusion (efficient-sid).

A diffusion prior fit to a *single* image, where the denoiser is available in closed form: the
optimal denoiser for a patch dataset is a softmax-weighted average over that image's own patches.
Nothing is trained -- the image *is* the model.

    from efficient_sid.image_denoiser import ImageDenoiser, extract_patches
    from efficient_sid.closed_form_denoiser import make_closed_form_denoiser
    from efficient_sid.pyramid import PyramidProcessor
    from efficient_sid.applications.uncond import sample_uncond

The six applications live in the ``efficient_sid.applications`` subpackage and are thin functions
over these pieces: unconditional sampling, retargeting, symmetry, tiling, structural analogy and
text-driven style. ``uncond`` is the coarse-to-fine loop the other five vary.
"""

__version__ = "0.1.0"
