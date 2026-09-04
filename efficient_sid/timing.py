"""Wall-time accounting for a run.

A single end-to-end number badly misrepresents this method: on a megapixel latent run most of the
clock is one-time setup (loading the VAE or CLIP) and file I/O, while the actual denoising is well
under a second. This module separates those out.

Design: nothing is instrumented at the call site.
- One-shot functions time themselves with the ``@timed`` / ``@timed_compute`` decorators
  (FluxVAE.from_pretrained / .encode / ._decode, invert_to_noise, get_clip_extractor, each sample*()).
- Hot paths (the denoiser, called hundreds of times; Laplacian blending) use event-based
  ``GPUTime`` -- a ``synchronize()`` per call would both slow them down and distort the timing.
A CLI constructs a ``StageTimer`` and calls ``report_run``.
"""

import functools
import time
import weakref
from collections import Counter
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Sequence

import torch

if TYPE_CHECKING:
    from efficient_sid.image_denoiser import ImageDenoiser


# ---------------------------------------------------------------------------
# Event-based per-component timing (for hot paths)
# ---------------------------------------------------------------------------

#: Every live ``GPUTime``, for ``drain_all``. Weak, so it does not keep a denoiser alive.
_GPU_TIMERS: "weakref.WeakSet" = weakref.WeakSet()


def drain_all() -> None:
    """Convert every timer's pending CUDA events into milliseconds. Totals are unchanged.

    ``record`` frees its two event handles only when the total is read, so a process drawing many
    samples would hold them all until the end.
    """
    if not torch.cuda.is_available():
        return
    pending = [t for t in _GPU_TIMERS if t._pairs]
    if pending:
        torch.cuda.synchronize()
        for t in pending:
            t._drain()


class GPUTime:
    """Accumulates GPU time for one component, without stalling the loop it runs in.

    A ``torch.cuda.synchronize()`` around every denoiser call would both slow the run down and
    distort what it is measuring (it blocks the CPU from running ahead). CUDA *events* are
    recorded asynchronously into the stream instead, and only resolved -- once -- when the
    total is read at report time.

    Components own an instance and time themselves, so the sampling loops need no instrumentation.
    """

    def __init__(self) -> None:
        self._pairs = []
        self._total_ms = 0.0
        self.calls = 0
        _GPU_TIMERS.add(self)

    @contextmanager
    def record(self) -> Iterator[None]:
        if not torch.cuda.is_available():
            t = time.perf_counter()
            yield
            self._total_ms += (time.perf_counter() - t) * 1e3
            self.calls += 1
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pairs.append((start, end))
            self.calls += 1

    def _drain(self) -> None:
        """Add the recorded events to the total. The caller must synchronize first."""
        for start, end in self._pairs:
            self._total_ms += start.elapsed_time(end)
        self._pairs = []

    @property
    def seconds(self) -> float:
        """Resolve any outstanding events (one sync) and return the accumulated time."""
        if self._pairs:
            torch.cuda.synchronize()
            self._drain()
        return self._total_ms / 1e3

    def reset(self) -> None:
        self._pairs, self._total_ms, self.calls = [], 0.0, 0


class WallTime:
    """Wall-clock counterpart to ``GPUTime``, with the same ``.record()`` / ``.seconds`` API.

    Use this when the work being timed is *not* GPU-kernel work -- e.g. a FAISS index build, which
    is CPU ``train``/``add`` plus host<->device transfer that CUDA events would miss. A component
    owns an instance and times itself with ``with self.<t>.record(): ...``, exactly like ``GPUTime``.
    """

    def __init__(self) -> None:
        self._total_ms = 0.0
        self.calls = 0

    @contextmanager
    def record(self) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self._total_ms += (time.perf_counter() - t) * 1e3
            self.calls += 1

    @property
    def seconds(self) -> float:
        return self._total_ms / 1e3

    def reset(self) -> None:
        self._total_ms, self.calls = 0.0, 0


# ---------------------------------------------------------------------------
# Decorator-based stage timing (for one-shot calls) + disk-I/O accounting
# ---------------------------------------------------------------------------

#: Stages recorded by the ``@timed`` decorator, in call order: [(label, seconds, nested)].
_TIMINGS = []

#: Time spent writing images to disk, so ``timed_compute`` can subtract it from a stage.
_IO = {"seconds": 0.0, "calls": 0}

#: Labels recorded by ``timed_compute`` -- the sampling stages. ``report_run`` puts setup above
#: them and their breakdown below.
_COMPUTE_LABELS = set()

#: Labels marked ``load=True``: reading model weights. Left out of core compute.
_LOAD_LABELS = set()

#: Laplacian-blending GPU time, aggregated across scales. ``PyramidProcessor`` records into it
#: and ``StageTimer`` reports it as a sub-line. Defined here rather than in ``pyramid``, which
#: imports this module.
BLEND_TIME = GPUTime()


def reset_timings() -> None:
    """Clear the recorded stages. Call between samples in a long-lived process (e.g. a demo)."""
    _TIMINGS.clear()
    _COMPUTE_LABELS.clear()
    _LOAD_LABELS.clear()
    _IO["seconds"], _IO["calls"] = 0.0, 0


@contextmanager
def record_io() -> Iterator[None]:
    """Charge the wrapped disk write to the separate I/O counter (see ``_IO``). Used by
    ``utils.save_image`` so image writing is reported on its own, never inside a compute stage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    try:
        yield
    finally:
        _IO["seconds"] += time.perf_counter() - t
        _IO["calls"] += 1


def timed(label: str, nested: bool = False, load: bool = False) -> Callable[[Callable], Callable]:
    """Decorate a one-shot function so it records its own wall time.

    Only for functions called a handful of times per run: it synchronizes CUDA on entry and exit,
    which is free once but would stall a hot loop. The denoiser is called hundreds of times, so it
    uses event-based ``GPUTime`` instead.

    ``nested=True`` marks the stage as a breakdown of another (e.g. decode happens *inside*
    sampling), so it is displayed but not double-counted in the total.

    ``load=True`` marks the stage as reading model weights: still a row of its own, but left out
    of core compute.
    """
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if load:
                    _LOAD_LABELS.add(label)
                _TIMINGS.append((label, time.perf_counter() - t, nested))
        return wrapper
    return decorate


def timed_compute(label: str) -> Callable[[Callable], Callable]:
    """Like ``timed``, but subtracts any image-writing done inside the call.

    The samplers write the final image (and, when intermediates are on, one per scale) from
    *within* the sampling call. Charging that to the method would inflate its cost, and make it
    look slower the more output you asked for.
    """
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            io_before = _IO["seconds"]
            try:
                return fn(*args, **kwargs)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                drain_all()
                wall = time.perf_counter() - t
                _COMPUTE_LABELS.add(label)
                _TIMINGS.append((label, wall - (_IO["seconds"] - io_before), False))
        return wrapper
    return decorate


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _collapse(stages: Sequence[Any]) -> list:
    """Merge stages sharing a label into one row: the run total, counted and averaged in the
    label. Each sampling call records its own entry, so N samples would print N identical rows.

    The number stays a total, so the counted column still sums to the run.
    """
    order, totals, counts, nested = [], {}, {}, {}
    for label, secs, is_nested in stages:
        if label not in totals:
            order.append(label)
            totals[label], counts[label], nested[label] = 0.0, 0, is_nested
        totals[label] += secs
        counts[label] += 1
    return [(label if counts[label] == 1
             else f"{label} (\u00d7{counts[label]}, mean {totals[label] / counts[label]:.2f}s)",
             totals[label], nested[label])
            for label in order]


class StageTimer:
    """Wall-time breakdown of a run, printed at the end.

    Most of a run's wall clock is one-time setup -- loading the VAE or CLIP -- not sampling, so a
    single end-to-end number badly misrepresents how fast the method is.

    The stages themselves are collected by the ``@timed`` decorator on the relevant functions and
    by the denoisers' own ``GPUTime``; a CLI only has to construct this and call ``report_run``.
    """

    def __init__(self, enabled: bool = True, t0: Optional[float] = None) -> None:
        self.enabled = enabled
        self.stages = []
        # t0 is captured by the CLI *before* `import torch`, so the reported TOTAL matches the
        # wall clock the user actually sees. Importing torch/diffusers can take tens of seconds
        # on a network filesystem, and a TOTAL that quietly excluded it would be unbelievable.
        self._t0 = t0 if t0 is not None else time.perf_counter()

    def add(self, label: str, seconds: float, nested: bool = False) -> None:
        if self.enabled:
            self.stages.append((label, seconds, nested))

    def report_run(self, title: str, denoisers: Optional[Sequence['ImageDenoiser']] = None) -> None:
        """Print the breakdown, pulling the self-timed components in.

        Counted stages are wall-clock and disjoint. Rows marked nested are a *breakdown of* an
        already-counted stage (inversion and decode both happen inside sampling), so they are
        displayed but not double-counted. The denoiser rows are pure GPU time -- the gap between
        them and the sampling stage is the DDIM step, the Laplacian blending, and saving.

        Note the inversion row *overlaps* the denoiser rows for apps that invert with the same
        denoisers they sample with (text_style); its label says so.

        Rows are emitted in the order the run happened: loading and encoding, the one-time index
        build, sampling with its breakdown, then output. ``_TIMINGS`` cannot supply that order --
        a nested stage is recorded before the stage containing it, and the index build is not
        recorded there at all.
        """
        if not self.enabled:
            return
        recorded = list(_TIMINGS) + self.stages
        setup = [s for s in recorded if not s[2] and s[0] not in _COMPUTE_LABELS]
        sampling = [s for s in recorded if not s[2] and s[0] in _COMPUTE_LABELS]
        inside_sampling = [s for s in recorded if s[2]]     # inversion, VAE decode

        self.stages = setup
        live = [(s, d) for s, d in enumerate(denoisers) if d is not None] if denoisers else []

        # ANN FAISS index build: a one-time per-scale setup cost (CPU train + host<->device
        # transfer), 0 for the exact backends. Built before any sampling, and amortized away when
        # the same image is reused across generations, so it gets a counted row above sampling.
        index_build = [(s, getattr(d.patch_denoiser, "index_build_time", None)) for s, d in live]
        total_build = sum(t.seconds for _, t in index_build if t is not None)
        if total_build > 0:
            self.add("ANN index build (one-time)", total_build)
            for s, t in index_build:
                if t is not None and t.seconds > 0:
                    self.add(f"index build: scale {s}", t.seconds, nested=True)

        self.stages += _collapse(sampling) + _collapse(inside_sampling)
        # The GPU rows below sum over the whole run, so above one sample the label says how many.
        runs = max(Counter(s[0] for s in sampling).values(), default=1)
        span = "" if runs == 1 else f", {runs} samples"
        if live:
            total = sum(d.image_time.seconds for _, d in live)
            patch = sum(d.patch_time.seconds for _, d in live)
            self.add(f"denoiser, all scales (GPU{span})", total, nested=True)
            self.add("of which: patch kernel", patch, nested=True)
            self.add("of which: unfold/fold", total - patch, nested=True)
            for s, d in live:
                self.add(f"scale {s} ({d.image_time.calls} calls)", d.image_time.seconds, nested=True)
        if BLEND_TIME.calls:
            self.add(
                f"Laplacian blending ({BLEND_TIME.calls} calls)",
                BLEND_TIME.seconds,
                nested=True,
            )
        # Disk I/O, counted on its own rather than inside the sampling stage.
        if _IO["calls"]:
            files = _IO["calls"]
            self.add(f"image writing ({files} file{'s' if files > 1 else ''})", _IO["seconds"])
        self.report(title)

    def report(self, title: str = 'Run') -> None:
        if not self.enabled or not self.stages:
            return
        total = time.perf_counter() - self._t0
        # The footer labels can outrun every stage label, so they set the column width too.
        footer = ("core compute (method only)", "startup + input read", "TOTAL")
        width = max([len(lbl) for lbl, _, _ in self.stages] + [len(f) for f in footer]) + 4
        print(f"\n{'='*66}")
        print(f"{title} — wall-time breakdown")
        print(f"{'-'*66}")
        for label, secs, nested in self.stages:
            pct = 100 * secs / total if total else 0
            bar = "" if nested else "#" * max(1, round(28 * secs / total)) if total else ""
            name = f"  └ {label}" if nested else label
            print(f"  {name:<{width}} {secs:8.2f}s  {pct:5.1f}%  {bar}")
        counted = sum(s for _, s, nested in self.stages if not nested)
        # Core compute = the method's own work: VAE encode/decode, the index build, the
        # denoising loop and the blend. Model loading and image writing keep their rows above but
        # are left out here -- neither varies with the config this number exists to compare.
        core = sum(s for lbl, s, nested in self.stages
                   if not nested and lbl not in _LOAD_LABELS
                   and not lbl.startswith("image writing"))
        print(f"  {'-'*(width + 24)}")
        print(f"  {footer[0]:<{width}} {core:8.2f}s")
        # Residual = everything not in a counted stage: torch import, config, and the (untimed)
        # input-image read. Output writing is *not* here -- it is the separate "image writing" stage.
        print(f"  {footer[1]:<{width}} {total-counted:8.2f}s")
        print(f"  {footer[2]:<{width}} {total:8.2f}s")
        print(f"{'='*66}\n")
