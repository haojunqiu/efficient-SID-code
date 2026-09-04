"""The closed-form denoiser: the optimal denoiser for a fixed set of data points.

Given a dataset of N D-dimensional vectors and a noisy query, the clean estimate is a
Gaussian-kernel-weighted average over the dataset -- Nadaraya-Watson kernel regression, which is
the *closed-form* optimal denoiser for that empirical distribution. No training.

Four backends compute exactly this, trading accuracy for speed/memory as the dataset grows. They
form a 2x2 -- {exhaustive, ANN retrieval} x {matmul softmax, fused SDPA} -- under one abstract base,
with a shared FAISS layer for the two ANN variants (indentation = inheritance):

  ClosedFormDenoiser                    abstract: the dataset + forward(xt, alpha, sigma)
    ExactClosedFormDenoiser             exhaustive -- full N x M score matrix, matmul softmax
    ExactFlashAttnClosedFormDenoiser    exhaustive -- identical math via fused SDPA, no matrix
    _ANNClosedFormDenoiserBase          shared FAISS retrieval: index build/search/free_gpu/chunking
      ANNClosedFormDenoiser             k nearest neighbours -- matmul softmax
      ANNFlashAttnClosedFormDenoiser    the same k-NN candidate set -- fused SDPA

Nothing here is patch-specific: the dataset is just N x D vectors. ``image_denoiser.py``
extracts and folds the image patches around it.
"""

import gc
import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from efficient_sid.timing import WallTime
from efficient_sid.utils import torch_dtype

try:
    import faiss
    import faiss.contrib.torch_utils  # enable direct CUDA tensor search
    _faiss_ok = True
except ImportError:
    # An installed faiss whose shared libraries will not load -- one built against another CUDA,
    # say -- raises ImportError rather than ModuleNotFoundError. Both leave the ANN backends
    # unusable and nothing else.
    _faiss_ok = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ClosedFormDenoiserConfig:
    """What every backend takes.

    ``type`` and ``query_chunks`` are each a scalar or a list with one entry per pyramid scale,
    so neither can be typed: OmegaConf cannot express a union with a list.
    """
    type: Any = "exact"                 # one of CLOSED_FORM_DENOISERS, or "auto"
    bottleneck_dtype: str = "bfloat16"  # what the patch kernel computes in, not precision.dtype
    query_chunks: Any = 1               # chunks the queries are split into; trades time for memory


@dataclass
class AnnClosedFormDenoiserConfig(ClosedFormDenoiserConfig):
    """...and what the two retrieval backends add: the FAISS index they search.

    A scale running an exact backend ignores them.
    """
    k: int = 5                   # nearest neighbours each query keeps
    index_type: str = "ivf"      # one of ANN_INDEX_TYPES
    nlist: Optional[int] = None  # IVF lists; None = FAISS's own heuristic, sqrt of the dataset
    nprobe: int = 2              # lists probed per query, for the two IVF indexes
    pq_m: int = 16               # PQ sub-quantizers, "ivfpq" only; must divide D


#: The FAISS indexes ``build_index`` can build.
ANN_INDEX_TYPES = ("flat", "ivf", "ivfpq")


# ---------------------------------------------------------------------------
# Base: the shared dataset dimensions + the forward contract
# ---------------------------------------------------------------------------

class ClosedFormDenoiser(torch.nn.Module):
    """Abstract base for the closed-form denoiser backends.

    Kernel regression over a dataset of N points: a noisy query returns a Gaussian-weighted
    average of the dataset.

    The base owns what every backend shares: the ``bottleneck_dtype`` its compute runs in, and
    the dataset's dimensions ``N`` (data points) and ``D`` (raw flattened vector length, e.g.
    C*P*P for patches -- *not* the padded ``d_head`` the flash backends use internally).

    It records ``N``/``D`` but does not keep the dataset tensor: each backend stores its own
    representation instead (raw, padded, or FAISS-indexed).

    A backend must implement::

        forward(self, xt, alpha, sigma) -> Tensor   # same shape as xt

    where ``xt`` is (M, D) noisy queries and ``alpha``/``sigma`` are the noise level to denoise
    at: the kernel is ``exp(-||xt - alpha*y||**2 / (2*sigma**2))`` over the dataset ``y``.

    ``query_chunks`` splits the queries into that many chunks, trading time for peak memory. The
    computation is the same either way.
    """

    #: The config class this backend is written against: what ``config.load`` validates a written
    #: block with, and where a scale's unwritten settings get their defaults.
    config_cls = ClosedFormDenoiserConfig

    def __init__(
        self,
        config: ClosedFormDenoiserConfig,
        dataset: torch.Tensor,
    ) -> None:
        super().__init__()
        self.bottleneck_dtype = torch_dtype(config.bottleneck_dtype)
        self.query_chunks = config.query_chunks
        # D is one flattened data point, unpadded -- flash_attn pads to d_head instead.
        self.N, self.D = dataset.flatten(1).shape   # size only; the tensor is not retained here

    def forward(self, xt: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "A ClosedFormDenoiser backend must implement forward(xt, alpha, sigma).")


# ---------------------------------------------------------------------------
# Exact (matmul-based softmax kernel regression)
# ---------------------------------------------------------------------------

class ExactClosedFormDenoiser(ClosedFormDenoiser):
    """Exact kernel regression over the whole dataset: one GEMM, softmax, weighted sum."""

    def __init__(
        self,
        config: ClosedFormDenoiserConfig,
        dataset: torch.Tensor,
    ) -> None:
        super().__init__(config, dataset)
        self.register_buffer('dataset', dataset.flatten(1))  # (N, D)

    def forward(self, xt: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type='cuda', dtype=self.bottleneck_dtype):
            alpha_dataset = alpha * self.dataset
            alpha_dataset_norm2 = (alpha_dataset ** 2).sum(-1).unsqueeze(0)

            out = []
            for xt_chunk in torch.chunk(xt, self.query_chunks):
                # Softmax normalizes along the dataset axis, where ||xt||**2 is constant,
                # so it is left out -- equivalent to cdist(xt, alpha*y)**2 then softmax. See
                # the paper's supplementary, Eq. (S31).
                weights = F.softmax(
                    (alpha_dataset_norm2 - 2 * xt_chunk @ alpha_dataset.T)
                    * (-1.0 / (2.0 * sigma ** 2)),
                    dim=1)
                out.append(torch.einsum('bn...,n...->b...', weights, self.dataset))
            out = torch.cat(out, dim=0)
        return out


# ---------------------------------------------------------------------------
# SDPA exact Gaussian via the homogeneous-coordinate trick
# ---------------------------------------------------------------------------

class ExactFlashAttnClosedFormDenoiser(ClosedFormDenoiser):
    """Exact kernel regression over the whole dataset, through PyTorch's fused SDPA
    (memory-efficient backend) via the homogeneous-coordinate trick.

    Identical to ``ExactClosedFormDenoiser`` and never materializes the N x M score
    matrix; the paper's supplementary S3.2 proves the equivalence.
    """

    def __init__(
        self,
        config: ClosedFormDenoiserConfig,
        dataset: torch.Tensor,
    ) -> None:
        super().__init__(config, dataset)

        dataset_flat = dataset.flatten(1)  # (N, D)
        D = self.D
        self.H = 1

        def roundup(x: int, tile: int) -> int:
            return ((x + tile - 1) // tile) * tile

        d_head_min = math.ceil((D + 1) / self.H)
        self.d_head = roundup(d_head_min, 64)
        self.pad_feat = self.d_head - D

        dataset_pad = F.pad(dataset_flat, (0, self.pad_feat))
        dataset_pad = dataset_pad.to(self.bottleneck_dtype).contiguous()
        self.register_buffer('dataset_pad', dataset_pad)
        self.register_buffer('dataset_norm2', (dataset_pad ** 2).sum(dim=-1))

    @torch.inference_mode()
    def forward(self, xt: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        B, D = xt.shape
        assert D == self.D
        device = xt.device

        q_pad = torch.zeros(B, self.d_head, device=device, dtype=self.dataset_pad.dtype)
        q_pad[:, :D] = xt
        q_pad[:, -1] = 1.0
        q = q_pad.view(1, B, self.H, self.d_head)

        k_base = self.dataset_pad * alpha
        bias = -0.5 * self.dataset_norm2 * (alpha ** 2)
        k_base[:, -1] = bias
        k = k_base.view(1, self.N, self.H, self.d_head)

        v = self.dataset_pad.view(1, self.N, self.H, self.d_head)

        scale = 1.0 / (sigma ** 2)

        out_lst = []
        q_chunks = torch.chunk(q, chunks=self.query_chunks, dim=1)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            for q_c in q_chunks:
                out_lst.append(
                    F.scaled_dot_product_attention(
                        q_c.permute(0, 2, 1, 3),
                        k.permute(0, 2, 1, 3),
                        v.permute(0, 2, 1, 3),
                        None,
                        0.0,
                        scale=scale,
                    ).permute(0, 2, 1, 3).view(1, q_c.shape[1], 1, self.d_head)
                )

        out = torch.cat(out_lst, dim=1)
        out = out.squeeze(1).reshape(B, self.d_head)
        return out[:, :D]


# ---------------------------------------------------------------------------
# ANN (FAISS) kernel regression
# ---------------------------------------------------------------------------

class _ANNClosedFormDenoiserBase(ClosedFormDenoiser):
    """Shared FAISS-retrieval machinery for the ANN backends. Not instantiated directly.

    Owns the FAISS GPU index (build + search), the neighbour count ``k``, the GPU lifecycle
    (``free_gpu``), and the query-chunked ``forward`` loop. A subclass supplies only its data
    representation (``_build_buffers``) and its per-chunk kernel-regression compute
    (``_forward_chunk``) -- the two things the ANN backends genuinely differ in.

    ``index_type`` / ``nlist`` / ``nprobe`` / ``pq_m`` are build-only: consumed by ``build_index``
    and not stored (``nprobe`` ends up on the FAISS index). Only ``k`` is kept -- it is read on
    every search.

    The index is expensive to build and only pays off when that cost is amortized over many
    searches, so it is built once, over the raw data points. But the kernel compares ``xt`` against
    ``alpha*y``, and ``alpha`` changes at every step -- so the scaling moves onto the query
    instead::

        ||xt - alpha*y||**2 = alpha**2 * ||xt/alpha - y||**2

    ``alpha**2`` does not depend on ``y``, so it cannot reorder the data points: both forms
    retrieve the same k nearest neighbours. Retrieval only ranks -- the weights are computed
    against ``alpha*y`` afterwards.

    ``ivf`` and ``ivfpq`` compare the query against k-means centroids to pick which lists to
    scan, and the identity applies there too. k-means is scale-equivariant -- it minimises
    squared distance, which ``alpha**2`` scales uniformly, leaving the same partition -- so
    clustering ``alpha*y`` gives centroids ``alpha*c`` over the same members, and ranking those
    against ``xt`` is ranking ``c`` against ``xt/alpha``. Both pick the same lists.
    """

    config_cls = AnnClosedFormDenoiserConfig

    def __init__(
        self,
        config: AnnClosedFormDenoiserConfig,
        dataset: torch.Tensor,
    ) -> None:
        if not _faiss_ok:
            raise RuntimeError(
                "faiss is required for the ANN backends; install faiss-gpu, or set "
                "image_denoiser.patch_denoiser.type to 'exact'.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required: the FAISS index is built on the GPU, as is every "
                "other backend. torch.cuda.is_available() is False here.")
        super().__init__(config, dataset)
        index_type, nlist = config.index_type, config.nlist
        pq_m, nprobe = config.pq_m, config.nprobe
        # Only "ivfpq" reads pq_m. Compared against the class default, not merely against
        # being inert: every ivf preset carries pq_m at its default, and warning on those
        # would fire on every ANN run. Same test as build.resolve's ``is_set``.
        if index_type.lower() != "ivfpq" and pq_m != AnnClosedFormDenoiserConfig.pq_m:
            warnings.warn(
                f"pq_m={pq_m} is set but index_type is {index_type!r}, so pq_m is ignored; "
                f"it applies to 'ivfpq' only.")
        self.device = torch.device("cuda")
        self.k = config.k
        data_flat = dataset.flatten(1)
        self._build_buffers(data_flat)
        #: x_hat at alpha = 0, where the kernel is flat and every data point weighs the same.
        #: Accumulated in float32: a bf16 dataset of millions of rows drifts otherwise.
        self.register_buffer('dataset_mean', data_flat.mean(0, dtype=torch.float32))
        #: Wall time to build the FAISS index (CPU train/add + host<->device transfer, which GPU
        #: events would miss). A one-time per-scale setup cost, amortized away when the same input
        #: image is reused across many generations; StageTimer reports it in its own row.
        self.index_build_time = WallTime()
        with self.index_build_time.record():
            self.faiss_index, self.gpu_res = self.build_index(
                data_flat,
                index_type=index_type,
                nlist=nlist,
                pq_m=pq_m,
                nprobe=nprobe,
            )
            torch.cuda.synchronize()

    # --- subclass hooks: the only per-backend differences --------------------
    def _build_buffers(self, data_flat: torch.Tensor) -> None:
        raise NotImplementedError("An ANN backend must implement _build_buffers(data_flat).")

    def _forward_chunk(
        self,
        xt_chunk: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "An ANN backend must implement _forward_chunk(xt_chunk, alpha, sigma).")

    # --- shared FAISS surface ------------------------------------------------
    @staticmethod
    def build_index(
        data_flat: torch.Tensor,
        index_type: str,
        nlist: Optional[int],
        pq_m: int,
        nprobe: int,
    ) -> Tuple[Any, Any]:
        """Build a FAISS GPU index over the (N, D) data point rows; return ``(gpu_index, gpu_res)``.

        A static method because it depends only on its arguments, not on instance state; both ANN
        backends call it from ``__init__``. It handles the ``nlist`` default (sqrt N), the ``ivfpq``
        ``pq_m``-divisor fallback, and a fallback to a flat index if training fails.

        ``gpu_res`` must be retained by the caller: the GPU index holds a reference to it, and
        ``free_gpu`` releases both.
        """
        cpu_data = data_flat.cpu().to(torch.float32).numpy()
        N, D = cpu_data.shape
        index_type = index_type.lower()
        if nlist is None:
            nlist = int(N ** 0.5)
        if index_type == "ivfpq" and D % pq_m != 0:
            divisors = [d for d in range(1, pq_m + 1) if D % d == 0]
            if not divisors:
                warnings.warn(f"Invalid pq_m={pq_m}; falling back to flat index")
                index_type = "flat"
            else:
                pq_m = max(divisors)

        if index_type == "flat":
            cpu_index = faiss.IndexFlatL2(D)
        elif index_type == "ivf":
            cpu_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, nlist, faiss.METRIC_L2)
        elif index_type == "ivfpq":
            cpu_index = faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, nlist, pq_m, 8, faiss.METRIC_L2)
        else:
            raise ValueError(f"Unknown index_type {index_type!r}; expected one of "
                             f"{', '.join(ANN_INDEX_TYPES)}")

        try:
            if hasattr(cpu_index, "is_trained") and not cpu_index.is_trained:
                cpu_index.train(cpu_data)
        except Exception as e:
            warnings.warn(f"FAISS training failed ({e}); falling back to flat index")
            cpu_index = faiss.IndexFlatL2(D)
        finally:
            cpu_index.add(cpu_data)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        gpu_res = faiss.StandardGpuResources()
        gpu_opts = faiss.GpuClonerOptions()
        gpu_opts.useFloat16 = False
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index, gpu_opts)
        if isinstance(cpu_index, faiss.IndexIVF):     # ivf / ivfpq -- not a flat fallback
            gpu_index.nprobe = nprobe
        return gpu_index, gpu_res

    @torch.no_grad()
    def _search_index(self, queries: torch.Tensor, k_eff: int) -> torch.Tensor:
        """Return the k nearest neighbours to each query.

        FAISS returns -1 where it found no neighbour, which torch would read as an index
        from the end of the dataset, so it is rejected here rather than gathered.
        """
        _, I = self.faiss_index.search(queries.to(torch.float32), k_eff)
        if (I < 0).any():
            raise RuntimeError(
                f"FAISS found fewer than {k_eff} neighbours for "
                f"{int((I < 0).any(dim=1).sum())} of {len(I)} queries. Raise nprobe, or "
                f"lower k below the smallest inverted list.")
        return I

    @torch.inference_mode()
    def forward(self, xt: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if alpha == 0:
            # The kernel weighs every data point equally, so x_hat is the mean for any query.
            # Retrieval cannot run either: the queries are scaled by 1/alpha.
            return self.dataset_mean.to(xt.dtype).expand_as(xt).contiguous()
        return torch.cat(
            [self._forward_chunk(chunk, alpha, sigma)
             for chunk in torch.chunk(xt, chunks=self.query_chunks)],
            dim=0,
        )

    def free_gpu(self) -> None:
        self.to('cpu')
        if getattr(self, 'faiss_index', None) is not None:
            del self.faiss_index
            self.faiss_index = None
        if getattr(self, 'gpu_res', None) is not None:
            del self.gpu_res
            self.gpu_res = None
        gc.collect()
        torch.cuda.empty_cache()


class ANNClosedFormDenoiser(_ANNClosedFormDenoiserBase):
    """ANN retrieval + explicit GEMM squared-L2 -> softmax -> weighted sum (bmm)."""

    def __init__(
        self,
        config: AnnClosedFormDenoiserConfig,
        dataset: torch.Tensor,
    ) -> None:
        super().__init__(config, dataset)

    def _build_buffers(self, data_flat: torch.Tensor) -> None:
        self.register_buffer('dataset', data_flat)   # raw data points, gathered per search

    def _forward_chunk(
        self,
        xt_chunk: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        B, D = xt_chunk.shape
        queries = (xt_chunk / alpha).contiguous()

        k_eff = min(self.k, self.N)
        idx = self._search_index(queries, k_eff)

        sel = self.dataset[idx.view(-1)].view(B, k_eff, -1)
        sel_scaled = alpha * sel

        def batched_sq_l2_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Compute row-wise squared L2 between each query x and its k neighbours y, via a GEMM."""
            x_norm = (x * x).sum(dim=-1, keepdim=True)
            y_norm = (y * y).sum(dim=-1)
            cross = (y @ x.unsqueeze(-1)).squeeze(-1)
            return (x_norm + y_norm - 2.0 * cross).clamp_min_(0.0)

        with torch.autocast(device_type='cuda', dtype=self.bottleneck_dtype):
            sq = batched_sq_l2_gemm(xt_chunk, sel_scaled)
            w = F.softmax(sq * (-1.0 / (2.0 * sigma ** 2)), dim=1)
            out = torch.bmm(w.unsqueeze(1), sel).squeeze(1)
        return out


# ---------------------------------------------------------------------------
# ANN + Flash-Attention hybrid
# ---------------------------------------------------------------------------

class ANNFlashAttnClosedFormDenoiser(_ANNClosedFormDenoiserBase):
    """ANN retrieval + PyTorch's fused SDPA (memory-efficient backend) via the
    homogeneous-coordinate trick."""

    def __init__(
        self,
        config: AnnClosedFormDenoiserConfig,
        dataset: torch.Tensor,
    ) -> None:
        super().__init__(config, dataset)

    def _build_buffers(self, data_flat: torch.Tensor) -> None:
        data_flat = data_flat.to(torch.float32)
        D = self.D

        tile = 64
        d_head_min = math.ceil((D + 1) / 1)
        self.d_head = ((d_head_min + tile - 1) // tile) * tile
        self.pad_feat = self.d_head - D

        data_pad_bf16 = F.pad(data_flat, (0, self.pad_feat)).to(self.bottleneck_dtype).contiguous()

        # Search reads exactly two things: the padded bf16 dataset (used as both K and V by SDPA)
        # and the un-padded squared norms. Norms are computed in fp32 before the cast, since
        # summing D squared terms in bf16 loses accuracy.
        self.register_buffer("dataset_pad", data_pad_bf16)
        self.register_buffer("dataset_norm2", (data_flat ** 2).sum(-1))

    def _forward_chunk(
        self,
        xt_chunk: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        B = xt_chunk.size(0)
        D = self.D
        scale = 1.0 / (sigma ** 2)

        k_eff = min(self.k, self.N)
        idx = self._search_index((xt_chunk / alpha).contiguous(), k_eff)

        neigh_bf16 = self.dataset_pad[idx.view(-1)].view(B, k_eff, self.d_head)
        norm2 = self.dataset_norm2[idx.view(-1)]
        bias_fp32 = -0.5 * norm2 * (alpha ** 2)

        # Keys carry the alpha scaling and the -|alpha*y|^2/2 bias, which together turn the dot
        # product into the negative squared distance. Values stay the raw patches: the estimate
        # is a weighted average of patches, not of scaled ones. K must therefore be a new tensor
        # -- an in-place `K *= alpha` would scale V through the alias and return alpha * x_hat.
        K_bf16 = neigh_bf16 * alpha
        K_bf16[..., -1] = bias_fp32.to(self.bottleneck_dtype).view(B, k_eff)
        V_bf16 = neigh_bf16

        Q_bf16 = torch.zeros(B, self.d_head, dtype=self.bottleneck_dtype, device=xt_chunk.device)
        Q_bf16[..., :D] = xt_chunk.to(self.bottleneck_dtype)
        Q_bf16[..., -1] = 1.0
        Q_bf16 = Q_bf16.view(B, 1, 1, self.d_head)

        K_bf16 = K_bf16.view(B, 1, k_eff, self.d_head)
        V_bf16 = V_bf16.view(B, 1, k_eff, self.d_head)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(
                Q_bf16,
                K_bf16,
                V_bf16,
                None,
                0.0,
                scale=scale,
            )

        out = out.view(B, self.d_head)[..., :D]
        return out.to(torch.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Every backend type, and the class that implements it.
CLOSED_FORM_DENOISERS = {
    "exact":            ExactClosedFormDenoiser,
    "exact_flash_attn": ExactFlashAttnClosedFormDenoiser,
    "ann":              ANNClosedFormDenoiser,
    "ann_flash_attn":   ANNFlashAttnClosedFormDenoiser,
}

#: The config class each type is written against, taken from the class that implements it.
CLOSED_FORM_DENOISER_CONFIGS = {t: cls.config_cls for t, cls in CLOSED_FORM_DENOISERS.items()}


def make_closed_form_denoiser(
    closed_form_denoiser_config: ClosedFormDenoiserConfig,
    dataset: torch.Tensor,
) -> ClosedFormDenoiser:
    """Build the backend ``closed_form_denoiser_config.type`` names, over ``dataset``.

    The config is one scale's, so ``type`` and ``query_chunks`` are scalars here. The ANN
    settings are read only by the two retrieval backends.
    """
    if closed_form_denoiser_config.type == "auto":
        raise ValueError("Denoiser type is still 'auto'. build.resolve picks the backend from the "
                         "image size; run the config through it first.")
    if not isinstance(closed_form_denoiser_config.type, str):
        raise ValueError(f"Denoiser type {closed_form_denoiser_config.type!r} holds one entry per scale; "
                         "build.image_denoiser_config_at_scale(...).patch_denoiser gives one scale's.")
    if closed_form_denoiser_config.type not in CLOSED_FORM_DENOISERS:
        raise ValueError(f"Unknown denoiser type {closed_form_denoiser_config.type!r}; expected one of "
                         f"{', '.join(CLOSED_FORM_DENOISERS)}")
    return CLOSED_FORM_DENOISERS[closed_form_denoiser_config.type](
        closed_form_denoiser_config, dataset)
