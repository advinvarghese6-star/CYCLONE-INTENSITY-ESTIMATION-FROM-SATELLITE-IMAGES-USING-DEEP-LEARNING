# Phase 3 — IMD categories, Grad-CAM, Gradio demo (fixed version, built on baseline_cnn_v3.pt)

GLM's Phase 3 notebook with these changes applied (everything else is untouched):
1. Cell 5: `%%writefile` path unquoted (it has no spaces).
2. Cell 9: the strong-end / weak-end readout now shows n.
3. Cell 10: Grad-CAM frame selection always includes the strongest populated category
   plus exactly 2 largest-error frames; the heatmap "observation" sentence now follows the
   computed overlap number instead of asserting a conclusion in advance.
4. Cell 11: added an end-to-end VALID-upload test above the UI block (a good Kelvin upload
   must reproduce the gallery path's answer).
5. Cell 11: Grad-CAM output switched from gr.Image to gr.Plot (Matplotlib figure), because the
   gr.Image output did not display in Colab; checked working end to end with an uploaded frame.
6. Cell 11: disclaimer now also states the strong-storm underestimation found in the per-category table.
7. Cell 13 (new, optional): demo-day launch with Drive unmounted.

Paste each cell into its own Colab cell, in order. Cells 7 and 8 are stop-gates.

## Cell 1 — defensive Drive mount + phase3 output dir
```python
# Defensive mount (Shared Context pattern): never `rm -rf` through a possibly-live
# mount; only clear /content/drive after the specific "must not already contain
# files" error.
from google.colab import drive
import shutil, os

try:
    drive.mount("/content/drive", force_remount=True)
except ValueError as e:
    if "must not already contain files" in str(e):
        shutil.rmtree("/content/drive", ignore_errors=True)
        drive.mount("/content/drive", force_remount=True)
    else:
        raise

DRIVE_PHASE1 = "/content/drive/MyDrive/TCIR/phase1"
DRIVE_PHASE2 = "/content/drive/MyDrive/TCIR/phase2"
DRIVE_PHASE3 = "/content/drive/MyDrive/TCIR/phase3"
os.makedirs(DRIVE_PHASE3, exist_ok=True)      # must exist before anything is saved
print("[OK] Drive mounted; phase3 dir ready:", DRIVE_PHASE3, os.path.isdir(DRIVE_PHASE3))
```

## Cell 2 — imports, seed, artifact checks
```python
import os, sys, json, math, gc, time
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device: {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda"
         else "  — CPU: evaluation will be slow but works"))

# Required artifacts from earlier phases — read-only this phase
for p in (DRIVE_PHASE1, DRIVE_PHASE2):
    assert os.path.isdir(p), f"missing folder: {p}"
for f in ("images_ir1.npy", "labels.npy", "storm_ids.npy",
          "train_idx.npy", "val_idx.npy", "split_meta.json"):
    assert os.path.exists(f"{DRIVE_PHASE1}/{f}"), f"missing Phase 1 artifact: {f}"
CKPT_PATH = f"{DRIVE_PHASE2}/baseline_cnn_v3.pt"
assert os.path.exists(CKPT_PATH), f"missing v3 checkpoint: {CKPT_PATH}"
print("[OK] all Phase 1 artifacts + v3 checkpoint found (read-only this phase)")
```

## Cell 3 — split_meta gates + Phase 1 constants (from file, never hardcoded)
```python
with open(f"{DRIVE_PHASE1}/split_meta.json") as f:
    split_meta = json.load(f)
print("=== split_meta.json ===")
print(json.dumps(split_meta, indent=2))

norm_block = split_meta.get("normalization", {})
assert norm_block.get("method") == "percentile_clip_0.5_99.5", (
    f"normalization.method = {norm_block.get('method')!r} — expected "
    f"'percentile_clip_0.5_99.5'; the images may be the broken min-max version.")
assert split_meta.get("source_md5"), "source_md5 missing from split_meta.json"
print("\n[OK] normalization method + source_md5 verified.")

def find_named(obj, names, path=()):
    """Recursive exact-name search for numeric leaves. Returns [(path, value)]."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = path + (str(k),)
            if str(k) in names and isinstance(v, (int, float)) and not isinstance(v, bool):
                hits.append((p, float(v)))
            if isinstance(v, (dict, list)):
                hits.extend(find_named(v, names, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                hits.extend(find_named(v, names, path + (str(i),)))
    return hits

def read_constant(what, names):
    hits = find_named(split_meta, names)
    assert len(hits) == 1, (
        f"Could not locate a unique '{what}' in split_meta.json "
        f"(candidate key names {sorted(names)}; found {hits}).\n"
        f"Normalization block actually saved by Phase 1:\n"
        f"{json.dumps(norm_block, indent=2)}")
    print(f"  {what:16s} = {hits[0][1]:.4f}   (path: {'/'.join(hits[0][0])})")
    return hits[0][1]

print("\n=== Phase 1 constants — read from split_meta.json, never hardcoded ===")
norm_min       = read_constant("norm_min",       {"norm_min", "clip_min", "min"})
norm_max       = read_constant("norm_max",       {"norm_max", "clip_max", "max"})
nan_fill_value = read_constant("nan_fill_value", {"nan_fill_value", "nan_fill", "fill_value"})

# Value gates — literals appear ONLY here, inside assertions:
assert 193.0 < norm_min < 194.0, f"norm_min={norm_min} — expected ~193.52 K"
assert 302.0 < norm_max < 303.0, f"norm_max={norm_max} — expected ~302.42 K"
assert norm_max > norm_min
if not (100.0 < nan_fill_value < 400.0):
    print(f"[WARN] nan_fill_value={nan_fill_value} is outside 100-400 K — expected a "
          f"Kelvin-space fill; check Phase 1 semantics before trusting the Kelvin path.")
else:
    print("[OK] constants in expected ranges (norm_min ~193.52 K, norm_max ~302.42 K, "
          "nan_fill in Kelvin space)")
```

## Cell 4 — checkpoint: full structure print, stats discovery, identity gates
```python
CKPT_MTIME = os.path.getmtime(CKPT_PATH)   # re-checked in the summary: never modified
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

def describe_structure(obj, name="ckpt", indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and all(isinstance(v, torch.Tensor) for v in vals):
            print(f"{pad}{name}: state_dict — {len(obj)} tensors "
                  f"(first keys: {list(obj)[:2]} ... last: {list(obj)[-1]})")
            return
        print(f"{pad}{name}: dict ({len(obj)} keys)")
        for k, v in obj.items():
            describe_structure(v, str(k), indent + 1)
    elif isinstance(obj, (list, tuple)):
        if obj and all(isinstance(v, (int, float)) for v in obj):
            print(f"{pad}{name}: list[{len(obj)}] numbers, first={obj[0]:.4f}, last={obj[-1]:.4f}")
        else:
            print(f"{pad}{name}: list[{len(obj)}]")
            for i, v in enumerate(obj[:3]):
                describe_structure(v, f"[{i}]", indent + 1)
    elif isinstance(obj, torch.Tensor):
        print(f"{pad}{name}: tensor {tuple(obj.shape)} {obj.dtype}")
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray {obj.shape} {obj.dtype}")
    elif isinstance(obj, str):
        s = obj if len(obj) <= 70 else obj[:67] + "..."
        print(f"{pad}{name}: str = {s!r}")
    elif obj is None or isinstance(obj, (int, float, bool)):
        print(f"{pad}{name}: {type(obj).__name__} = {obj}")
    else:
        print(f"{pad}{name}: {type(obj).__name__}")

print("=== baseline_cnn_v3.pt — full structure, nested keys included ===")
describe_structure(ckpt)

# Known-facts gates — confirms this is the right checkpoint:
assert ckpt.get("best_epoch") == 21, f"best_epoch={ckpt.get('best_epoch')} — expected 21"
assert abs(float(ckpt["best_val_mae_knots"]) - 11.6101) < 1e-3, "best_val_mae_knots mismatch"
assert abs(float(ckpt["best_val_rmse_knots"]) - 16.6257) < 1e-3, "best_val_rmse_knots mismatch"
print("\n[OK] checkpoint identity: best_epoch 21, recorded val MAE 11.6101 / RMSE 16.6257 kn")

# --- locate the standardization stats WITHOUT trusting key names ----------
def find_stat(d, needle, exclude=("v2", "reference"), path=()):
    """Numeric leaves in the normalization block whose key contains `needle`,
    skipping v2's full-frame reference stats (kept in the ckpt for the record)."""
    hits = []
    for k, v in d.items():
        p = path + (str(k),)
        plow = "/".join(p).lower()
        if isinstance(v, dict):
            hits.extend(find_stat(v, needle, exclude, p))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and needle in str(k).lower():
            if not any(s in plow for s in exclude):
                hits.append((p, float(v)))
    return hits

norm_info = ckpt.get("normalization")
assert isinstance(norm_info, dict), (
    f"'normalization' block missing; top-level keys: {list(ckpt)}")
mean_hits = find_stat(norm_info, "mean")
std_hits  = find_stat(norm_info, "std")
print(f"\nmean candidates (non-reference): {mean_hits}")
print(f"std  candidates (non-reference): {std_hits}")
assert len(mean_hits) == 1 and len(std_hits) == 1, \
    "could not uniquely locate the v3 standardization stats in the checkpoint"
train_mean = mean_hits[0][1]
train_std  = std_hits[0][1]
print(f"\nstandardization stats (path {'/'.join(mean_hits[0][0])}): "
      f"train_mean={train_mean:.6f}, train_std={train_std:.6f}")
# value gates — literals in assertions only:
assert abs(train_mean - 0.4350) < 0.03, f"train_mean={train_mean} — expected ~0.4350"
assert abs(train_std - 0.2869) < 0.03, f"train_std={train_std} — expected ~0.2869"
print("[OK] stats match the recorded ~0.4350 / ~0.2869")

# --- crop window: canonical 201//2 ± 32, cross-checked against the ckpt ----
C0, C_LO, C_HI = 201 // 2, 201 // 2 - 32, 201 // 2 + 32      # 100, 68, 132
crop_block = ckpt.get("crop", {})
if "row_range" in crop_block:
    assert (list(crop_block["row_range"]) == [C_LO, C_HI]
            and list(crop_block["col_range"]) == [C_LO, C_HI]), (
        f"checkpoint crop block disagrees with canonical [68,132): {crop_block}")
print(f"[OK] crop window rows/cols [{C_LO}, {C_HI}) — 201//2 ± 32, matches ckpt crop block")
```

## Cell 5 — tc_inference.py (shared module, written to Drive)
```python
%%writefile /content/drive/MyDrive/TCIR/phase3/tc_inference.py
"""
tc_inference.py — Phase 3 shared inference module (single source of truth).

Everything that turns pixels into a v3-model number lives here: Phase 1/2
preprocessing (Kelvin -> [0,1] -> center crop -> standardize -> (1,64,64)),
single-pass and rotation-blended prediction, regression Grad-CAM, and the
IMD convenience category mapping. 3a, 3b and 3c all import THIS module so the
demo's preprocessing cannot drift from the evaluation pipeline.

No normalization constants are hardcoded here — the notebook builds a
TCPreprocessor from split_meta.json + baseline_cnn_v3.pt and passes it around.
"""
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# IMD convenience categories (knots), half-open intervals. This is a mapping
# onto TCIR best-track winds for analysis convenience — NOT an official IMD
# classification (agencies define/average sustained wind differently).
# ---------------------------------------------------------------------------
IMD_THRESHOLDS = np.array([17.0, 28.0, 34.0, 48.0, 64.0, 90.0, 120.0])
IMD_NAMES = [
    "Low Pressure Area", "Depression", "Deep Depression", "Cyclonic Storm",
    "Severe Cyclonic Storm", "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm", "Super Cyclonic Storm",
]
IMD_SHORT = ["Low", "Dep", "DeepDep", "CS", "SevCS", "VSevCS", "ExtSevCS", "Super"]


def imd_category_index(wind_knots):
    """Scalar wind (knots) -> category index 0..7. side='right' gives [t, t_next)."""
    return int(np.searchsorted(IMD_THRESHOLDS, float(wind_knots), side="right"))


def imd_category_indices(wind_knots):
    """Vectorized: array of winds -> int array of category indices (same shape)."""
    w = np.asarray(wind_knots, dtype=float)
    return np.searchsorted(IMD_THRESHOLDS, w, side="right").astype(int)


# ---------------------------------------------------------------------------
# Architecture — copied EXACTLY from Phase 2 v3 (state_dict keys depend on
# layer order/names). No custom init here: a fresh TCNet() has PyTorch's
# DEFAULT init, which is exactly what the randomized-weights check needs.
# ---------------------------------------------------------------------------
class TCNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Preprocessing. Training-time order (Phase 2 v3), exactly:
#   [0,1] 201x201 frame -> crop [68,132)x[68,132) -> (x-mean)/std -> (1,64,64)
# Kelvin frames first go through Phase 1's step: NaN-fill -> clip -> scale.
# All entry points compose the SAME primitives, so each step exists exactly
# once — that is the point of this module.
# ---------------------------------------------------------------------------
class TCPreprocessor:
    def __init__(self, norm_min, norm_max, nan_fill_value, train_mean, train_std,
                 crop_lo=68, crop_hi=132):
        self.norm_min = float(norm_min)
        self.norm_max = float(norm_max)
        self.nan_fill_value = float(nan_fill_value)
        self.train_mean = float(train_mean)
        self.train_std = float(train_std)
        self.crop_lo, self.crop_hi = int(crop_lo), int(crop_hi)
        assert self.norm_max > self.norm_min
        assert self.train_std > 0
        assert self.crop_hi - self.crop_lo == 64

    # ---- primitives --------------------------------------------------------
    def _check_frame(self, frame):
        a = np.asarray(frame)
        if a.ndim != 2 or a.shape != (201, 201):
            raise ValueError(f"expected a (201, 201) frame, got shape {a.shape}")
        return a

    def crop_unit(self, frame01):
        """[0,1] 201x201 -> 64x64 unit-space crop (display / shared step)."""
        a = self._check_frame(frame01)
        return np.ascontiguousarray(
            a[self.crop_lo:self.crop_hi, self.crop_lo:self.crop_hi],
            dtype=np.float32)

    def kelvin_to_unit(self, frame_kelvin):
        """Raw Kelvin 201x201 -> [0,1] 201x201 (Phase 1 order: NaN-fill, clip, scale)."""
        a = self._check_frame(frame_kelvin).astype(np.float64)
        a = np.where(np.isnan(a), self.nan_fill_value, a)
        a = np.clip(a, self.norm_min, self.norm_max)
        return (a - self.norm_min) / (self.norm_max - self.norm_min)

    def _standardize_reshape(self, unit_crop):
        # float32 arithmetic with mean/std rounded to float32 — bit-identical to
        # Phase 2 v3's ((x - mean) / std) on the float32 cropped frames.
        c = np.asarray(unit_crop, dtype=np.float32)
        z = (c - np.float32(self.train_mean)) / np.float32(self.train_std)
        return z.reshape(1, 64, 64)

    # ---- public entry points ------------------------------------------------
    def preprocess_unit(self, frame01):
        """Phase 1 [0,1] frame (201,201) -> standardized (1,64,64) float32."""
        return self._standardize_reshape(self.crop_unit(frame01))

    def preprocess_kelvin(self, frame_kelvin):
        """Raw IR1 Kelvin frame (201,201) -> standardized (1,64,64) float32."""
        return self._standardize_reshape(self.crop_unit(self.kelvin_to_unit(frame_kelvin)))

    def describe(self):
        return (f"TCPreprocessor(norm_min={self.norm_min:.4f} K, "
                f"norm_max={self.norm_max:.4f} K, nan_fill={self.nan_fill_value:.2f} K, "
                f"train_mean={self.train_mean:.6f}, train_std={self.train_std:.6f}, "
                f"crop=[{self.crop_lo},{self.crop_hi}))")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _as_batch(x_std):
    x = np.asarray(x_std, dtype=np.float32)
    if x.ndim == 3 and x.shape == (1, 64, 64):          # single frame -> batch of 1
        x = x[None]
    if x.ndim != 4 or x.shape[1:] != (1, 64, 64):
        raise ValueError(f"expected (B,1,64,64) standardized input, got {x.shape}")
    return np.ascontiguousarray(x)


def _forward(model, x_std):
    device = next(model.parameters()).device
    t = torch.from_numpy(np.ascontiguousarray(x_std, dtype=np.float32)).to(device)
    with torch.no_grad():
        return model(t).float().cpu().numpy()


def predict_single(model, x_std):
    """Single-pass prediction on the ORIGINAL orientation. Returns (B,) np."""
    assert not model.training, "call model.eval() before predicting"
    return _forward(model, _as_batch(x_std))


# Rotation-blend recipe that produced the Phase 2 reference numbers: 10 evenly
# spaced angles on the STANDARDIZED batch, axes=(2,3), order=1, constant fill
# 0.0 (the standardized-space neutral value), predictions averaged.
BLEND_ANGLES = np.linspace(0.0, 360.0, 10, endpoint=False)


def rotate_batch(x_std, angle):
    return scipy.ndimage.rotate(x_std, angle, axes=(2, 3), reshape=False,
                                order=1, mode="constant", cval=0.0)


def predict_blended(model, x_std, angles=None):
    """Rotation-blended + single-pass prediction.

    Returns dict:
      blended   — mean over the rotated predictions                     (B,)
      single    — plain forward on the unrotated frame                  (B,)
      spread    — std over the rotated predictions: agreement across
                  rotations, NOT a confidence interval                  (B,)
      per_angle — (A, B) individual rotated predictions
    """
    assert not model.training, "call model.eval() before predicting"
    x = _as_batch(x_std)
    angles = BLEND_ANGLES if angles is None else np.asarray(angles, dtype=float)
    per = np.stack([_forward(model, rotate_batch(x, float(a))) for a in angles], axis=0)
    return {
        "blended": per.mean(axis=0),
        "single": _forward(model, x),
        "spread": per.std(axis=0, ddof=0),
        "per_angle": per,
        "angles": angles,
    }


# ---------------------------------------------------------------------------
# Regression Grad-CAM. Requires ReLU(inplace=False) — with in-place ReLU the
# forward pass can overwrite the hooked tensor. BatchNorm stays in eval mode;
# gradients are ENABLED (no torch.no_grad) — we backprop from the scalar output.
# Caption to display with these maps: "regions where the output is most
# sensitive to evidence for HIGHER predicted wind speed".
# ---------------------------------------------------------------------------
def grad_cam(model, x_std):
    """Grad-CAM for ONE standardized frame, original orientation.

    x_std: (1,64,64) or (1,1,64,64) float32, standardized space.
    Returns (cam_display, cam_raw, prediction):
      cam_display — (64,64) normalized to [0,1] for DISPLAY ONLY
      cam_raw     — (64,64) upsampled map before normalization
      prediction  — the model's scalar output for this forward
    """
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            assert not m.inplace, ("grad_cam must not run on an in-place-ReLU model: "
                                   "the hooked tensor can be overwritten mid-forward")
    assert not model.training, "call model.eval() before grad_cam"
    device = next(model.parameters()).device
    x = np.asarray(x_std, dtype=np.float32)
    if x.ndim == 3 and x.shape == (1, 64, 64):
        x = x[None]
    assert x.shape == (1, 1, 64, 64), f"expected one (1,1,64,64) frame, got {x.shape}"
    x_t = torch.from_numpy(x).to(device)

    target = model.features[11]     # final ReLU before GAP; output shape (1,256,4,4)
    stash = {}
    hf = target.register_forward_hook(lambda m, i, o: stash.__setitem__("act", o))
    hb = target.register_full_backward_hook(lambda m, gi, go: stash.__setitem__("grad", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        out = model(x_t)            # (1,) — NO no_grad here: gradients required
        out.backward()
        A = stash["act"].detach()   # (1,256,4,4)
        G = stash["grad"].detach()  # (1,256,4,4)
        assert tuple(A.shape) == (1, 256, 4, 4), f"unexpected hook shape {tuple(A.shape)}"
        w = G.mean(dim=(2, 3))                                  # channel weights (1,256)
        cam = F.relu((w[:, :, None, None] * A).sum(dim=1))      # (1,4,4)
        cam_up = F.interpolate(cam.unsqueeze(1), size=(64, 64), mode="bilinear",
                               align_corners=False).squeeze(1)  # (1,64,64)
        cam_raw = cam_up.squeeze(0).cpu().numpy()
        pred = float(out.item())
    finally:
        hf.remove()
        hb.remove()
    m = float(cam_raw.max())
    cam_disp = cam_raw / m if m > 0 else np.zeros_like(cam_raw)
    return cam_disp, cam_raw, pred
```

## Cell 6 — import module, rebuild model (389,057), val-only mmap load, overlap gate
```python
import sys
sys.path.insert(0, DRIVE_PHASE3)
import tc_inference as tci
# If you edit the module and re-run Cell 5, re-import with:
# import importlib; tci = importlib.reload(tci)
print(f"imported shared module from: {tci.__file__}")

# ---- rebuild the model from the checkpoint (read-only; no training) --------
model = tci.TCNet(in_channels=1).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel rebuilt from {CKPT_PATH}")
print(f"Total parameters: {n_params:,}")
assert n_params == 389_057, f"param count {n_params:,} != expected 389,057 — architecture mismatch"
print("[OK] param count 389,057; model.eval() set")

preproc = tci.TCPreprocessor(norm_min, norm_max, nan_fill_value,
                             train_mean, train_std, C_LO, C_HI)
print(f"preprocessor: {preproc.describe()}")

# ---- load ONLY the val frames (mmap; the 18,899 train rows are never read) -
train_idx = np.load(f"{DRIVE_PHASE1}/train_idx.npy")
val_idx   = np.load(f"{DRIVE_PHASE1}/val_idx.npy")
labels    = np.load(f"{DRIVE_PHASE1}/labels.npy")
storm_ids = np.load(f"{DRIVE_PHASE1}/storm_ids.npy", allow_pickle=True)

images_mmap = np.load(f"{DRIVE_PHASE1}/images_ir1.npy", mmap_mode="r")
print(f"\nimages (memmap, lazy): shape={images_mmap.shape}, dtype={images_mmap.dtype}")
x_val01 = np.array(images_mmap[val_idx])    # fancy indexing copies ONLY these rows
del images_mmap
gc.collect()

y_val = labels[val_idx].astype(np.float32)
val_storm_ids = storm_ids[val_idx]
print(f"x_val01: {x_val01.shape} {x_val01.dtype} "
      f"({x_val01.nbytes/1e6:.1f} MB in RAM — train frames never touched)")
print(f"y_val  : {y_val.shape}, wind range {y_val.min():.0f}–{y_val.max():.0f} knots")
# Frames are Phase 1 unit-space [0,1] (NOT Kelvin — do not check 150–320 K here):
assert -0.01 <= x_val01.min() and x_val01.max() <= 1.01, "val frames outside [0,1]"

# ---- re-run the storm-level zero-overlap gate before analysing anything ----
assert set(train_idx.tolist()) & set(val_idx.tolist()) == set(), "frame-level overlap"
assert len(train_idx) + len(val_idx) == len(labels), "train+val != total frames"
train_storms = set(storm_ids[train_idx].tolist())
val_storms   = set(val_storm_ids.tolist())
overlap = train_storms & val_storms
assert len(overlap) == 0, (f"STORM-LEVEL OVERLAP: {sorted(overlap)[:5]} — group leakage; aborting")
print(f"[OK] zero storm overlap: train={len(train_storms)} storms, "
      f"val={len(val_storms)} storms, overlap=0")

# one-frame smoke before the heavy evaluation:
_p0 = float(tci.predict_single(model, preproc.preprocess_unit(x_val01[0]))[0])
print(f"\nsmoke: frame 0 → pred {_p0:.2f} kn, true {y_val[0]:.2f} kn")
```

## Cell 7 — parity test (stop-gate)
```python
print("=== Parity test: Kelvin path vs [0,1] path — one shared module, 20 val frames ===")
par_pos = np.unique(np.round(np.linspace(0, len(x_val01) - 1, 20)).astype(int))[:20]
max_diff = 0.0
for p in par_pos:
    frame01 = x_val01[p]
    a = preproc.preprocess_unit(frame01)                                # [0,1] path
    K = frame01.astype(np.float64) * (norm_max - norm_min) + norm_min   # back to Kelvin
    b = preproc.preprocess_kelvin(K)                                    # raw-Kelvin path
    max_diff = max(max_diff, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
print(f"frames checked: {len(par_pos)}   max |unit-path − kelvin-path| = {max_diff:.3e}")
PARITY_OK = max_diff < 1e-4
assert PARITY_OK, (f"Parity test FAILED: max diff {max_diff:.3e} ≥ 1e-4 — the Kelvin "
                   f"upload path disagrees with the evaluation path. Fix before the demo.")
print("[OK] parity test passed — demo uploads go through the identical pipeline")
```

## Cell 8 — reproduction test (stop-gate)
```python
print("=== Reproduction test: Phase 2 v3 reference numbers on the val set ===")
t0 = time.time()

def evaluate_all(frames01, batch_size=64):
    M = len(frames01)
    single, blended, spread = (np.empty(M, np.float64) for _ in range(3))
    for s in range(0, M, batch_size):                      # same batch size & order as Phase 2
        frames = frames01[s:s + batch_size]
        x_std = np.stack([preproc.preprocess_unit(f) for f in frames])   # (B,1,64,64)
        res = tci.predict_blended(model, x_std)
        single[s:s + len(frames)]  = res["single"]
        blended[s:s + len(frames)] = res["blended"]
        spread[s:s + len(frames)]  = res["spread"]
    return single, blended, spread

single_preds, blended_preds, spread_preds = evaluate_all(x_val01)
print(f"evaluated {len(y_val)} frames in {time.time() - t0:.1f}s "
      f"(batch 64, val_idx order — same as Phase 2's val_loader)")

err_s, err_b = single_preds - y_val, blended_preds - y_val
mae_s, rmse_s = float(np.abs(err_s).mean()), float(np.sqrt((err_s ** 2).mean()))
mae_b, rmse_b = float(np.abs(err_b).mean()), float(np.sqrt((err_b ** 2).mean()))

REF = {"mae_s": 11.6101, "rmse_s": 16.6257, "mae_b": 11.30, "rmse_b": 16.36}
print(f"\n{'':16s} {'MAE (kn)':>10s} {'RMSE (kn)':>10s}   reference        tol")
print(f"{'single-pass':16s} {mae_s:10.4f} {rmse_s:10.4f}   "
      f"{REF['mae_s']:.4f} / {REF['rmse_s']:.4f}   0.01")
print(f"{'blended (10)':16s} {mae_b:10.4f} {rmse_b:10.4f}   "
      f"{REF['mae_b']:.2f} / {REF['rmse_b']:.2f}     0.02")

REPRO_SINGLE_OK = (abs(mae_s - REF["mae_s"]) <= 0.01) and (abs(rmse_s - REF["rmse_s"]) <= 0.01)
REPRO_BLEND_OK  = (abs(mae_b - REF["mae_b"]) <= 0.02) and (abs(rmse_b - REF["rmse_b"]) <= 0.02)
print(f"single-pass: {'PASS' if REPRO_SINGLE_OK else 'FAIL'}   blended: {'PASS' if REPRO_BLEND_OK else 'FAIL'}")

if not (REPRO_SINGLE_OK and REPRO_BLEND_OK):
    print("\n" + "!" * 78)
    print("REPRODUCTION TEST FAILED — the numbers above do not match Phase 2's")
    print("reference. STOPPING HERE: do not continue to 3a/3b/3c and do not launch")
    print("the demo. Bring this printed output back before anything else.")
    print("!" * 78)
    raise RuntimeError(
        f"Reproduction mismatch — single {mae_s:.4f}/{rmse_s:.4f} vs "
        f"{REF['mae_s']}/{REF['rmse_s']}; blended {mae_b:.4f}/{rmse_b:.4f} vs "
        f"{REF['mae_b']}/{REF['rmse_b']}")
print("\n[OK] reproduction test passed — this pipeline IS the Phase 2 one")
```

## Cell 9 — 3a: IMD category analysis
```python
from sklearn.metrics import confusion_matrix

# ---- IMD mapping: boundary unit test --------------------------------------
_cases = [16.99, 17.0, 27.99, 28.0, 119.99, 120.0]
_exp   = [0, 1, 1, 2, 6, 7]
for w, e in zip(_cases, _exp):
    got = tci.imd_category_index(w)
    assert got == e, f"IMD boundary test failed: {w} → {tci.IMD_NAMES[got]} (expected {tci.IMD_NAMES[e]})"
assert list(tci.imd_category_indices(np.array(_cases))) == _exp
print("[OK] IMD boundary unit test:",
      ", ".join(f"{w}→{tci.IMD_NAMES[e]}" for w, e in zip(_cases, _exp)))

# ---- ONE helper, called twice ----------------------------------------------
def category_metrics(preds, y_true):
    preds, y_true = np.asarray(preds, float), np.asarray(y_true, float)
    t_idx, p_idx = tci.imd_category_indices(y_true), tci.imd_category_indices(preds)
    n_true = np.bincount(t_idx, minlength=8)
    cm = confusion_matrix(t_idx, p_idx, labels=list(range(8)))   # all 8 labels, always
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_row = np.where(n_true[:, None] > 0, cm / np.maximum(n_true[:, None], 1), np.nan)
    per = []
    for i in range(8):
        m, n = t_idx == i, int((t_idx == i).sum())
        e = preds[m] - y_true[m]
        per.append({"name": tci.IMD_NAMES[i], "n": n,
                    "mae": float(np.abs(e).mean()) if n else float("nan"),
                    "signed": float(e.mean()) if n else float("nan")})
    err = preds - y_true
    return {"n_true": n_true, "cm": cm, "cm_row": cm_row, "per": per,
            "exact_acc": float((t_idx == p_idx).mean()),
            "within1_acc": float((np.abs(t_idx - p_idx) <= 1).mean()),
            "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}

def print_category_report(rep, name):
    print(f"\n=== 3a — category report ({name}) ===")
    print("val frames per TRUE category:")
    for i in range(8):
        print(f"  {tci.IMD_NAMES[i]:34s} n={int(rep['n_true'][i]):4d}")
    print(f"\nexact-category accuracy      : {rep['exact_acc']:.3f}")
    print(f"within-one-category accuracy : {rep['within1_acc']:.3f}"
          f"   (|category(true) − category(pred)| ≤ 1)")
    print(f"overall MAE {rep['mae']:.2f} kn / RMSE {rep['rmse']:.2f} kn")
    print(f"\n{'true category':34s} {'n':>5s} {'MAE':>7s} {'signed err':>10s}")
    for r in rep["per"]:
        if r["n"] == 0:
            print(f"{r['name']:34s} {'n=0':>5s} {'—':>7s} {'—':>10s}")
        else:
            star = "  *" if r["n"] < 30 else ""
            print(f"{r['name']:34s} {r['n']:5d} {r['mae']:7.2f} {r['signed']:+10.2f}{star}")
    print("  (* n<30 — from very few storms; indicative only)")
    print("\nconfusion matrix — raw counts (rows TRUE, cols PREDICTED):")
    print(pd.DataFrame(rep["cm"], index=[f"T:{s}" for s in tci.IMD_SHORT],
                       columns=[f"P:{s}" for s in tci.IMD_SHORT]).to_string())
    print("\nconfusion matrix — row-normalized (— = empty true category):")
    print("        " + " ".join(f"{s:>8s}" for s in tci.IMD_SHORT))
    for i in range(8):
        cells = [f"{'—':>8s}" if np.isnan(rep["cm_row"][i, j])
                 else f"{rep['cm_row'][i, j]*100:7.1f}%" for j in range(8)]
        print(f"T:{tci.IMD_SHORT[i]:>6s} " + " ".join(cells))

rep_single  = category_metrics(single_preds, y_val)
rep_blended = category_metrics(blended_preds, y_val)
print_category_report(rep_single,  "single-pass predictions")
print_category_report(rep_blended, "rotation-blended predictions (10 angles)")

# ---- per-category table → CSV ----------------------------------------------
df_cat = pd.DataFrame([{
    "category": r["name"], "n": r["n"],
    "mae_single": rep_single["per"][i]["mae"],
    "signed_err_single": rep_single["per"][i]["signed"],
    "mae_blended": r["mae"], "signed_err_blended": r["signed"],
} for i, r in enumerate(rep_blended["per"])])
csv_path = f"{DRIVE_PHASE3}/category_table_v3.csv"
df_cat.to_csv(csv_path, index=False, na_rep="")
print(f"\nSaved {csv_path}")
print(df_cat.round(2).to_string(index=False))

# ---- predicted-vs-true scatter ----------------------------------------------
vmax_err = float(max(np.abs(err_s).max(), np.abs(err_b).max()))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True, sharey=True)
for ax, preds, ttl in [
        (axes[0], single_preds,
         f"single-pass (MAE {rep_single['mae']:.2f}, RMSE {rep_single['rmse']:.2f} kn)"),
        (axes[1], blended_preds,
         f"rotation-blended ×10 (MAE {rep_blended['mae']:.2f}, RMSE {rep_blended['rmse']:.2f} kn)")]:
    sc = ax.scatter(y_val, preds, s=9, alpha=0.45,
                    c=np.abs(preds - y_val), cmap="magma", vmin=0, vmax=vmax_err)
    lim = [0, float(max(y_val.max(), preds.max())) + 8]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    for t in tci.IMD_THRESHOLDS:                       # where category flips happen
        ax.axhline(t, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax.axvline(t, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true wind speed (knots)"); ax.set_ylabel("predicted (knots)")
    ax.set_title(ttl, fontsize=11); ax.legend(loc="upper left"); ax.grid(alpha=0.25)
fig.colorbar(sc, ax=axes, shrink=0.85, label="|error| (knots)")
fig.suptitle("Phase 3a — predicted vs true wind speed, val set, v3 checkpoint "
             "(dotted lines = IMD thresholds)", fontsize=12)
scatter_path = f"{DRIVE_PHASE3}/scatter_pred_vs_true_v3.png"
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
print(f"Saved {scatter_path}")
plt.show()

# ---- confusion-matrix PNGs (raw + row-normalized each) -----------------------
def plot_confusion(rep, name, path):
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.6))
    panels = [(axes[0], rep["cm"], "raw counts", False),
              (axes[1], rep["cm_row"], "row-normalized (share of true category)", True)]
    for ax, M, ttl, normalized in panels:
        vmax = float(np.nanmax(M)) if normalized else float(M.max())
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(8):
            for j in range(8):
                v = M[i, j]
                if np.isnan(v):
                    txt, color = "—", "0.4"
                elif normalized:
                    txt, color = f"{v*100:.0f}%", ("white" if v > 0.55 else "black")
                else:
                    txt, color = f"{int(v)}", ("white" if v > 0.55 * vmax else "black")
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks(range(8))
        ax.set_xticklabels(tci.IMD_SHORT, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(8)); ax.set_yticklabels(tci.IMD_SHORT, fontsize=9)
        ax.set_xlabel("predicted category"); ax.set_ylabel("true category")
        ax.set_title(ttl, fontsize=11)
    fig.suptitle(f"Phase 3a — IMD convenience categories, confusion matrix ({name})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.show()

plot_confusion(rep_single,  "single-pass",        f"{DRIVE_PHASE3}/cm_single_v3.png")
plot_confusion(rep_blended, "rotation-blended ×10", f"{DRIVE_PHASE3}/cm_blended_v3.png")

# ---- two-line plain-language readout (data-driven) ---------------------------
pop_rows = [r for r in rep_blended["per"] if r["n"] > 0]
worst2 = sorted(pop_rows, key=lambda r: -r["mae"])[:2]
strong_i = max(i for i in range(8) if rep_blended["per"][i]["n"] > 0)
weak_i   = min(i for i in range(8) if rep_blended["per"][i]["n"] > 0)
ss = rep_blended["per"][strong_i]["signed"]
sw = rep_blended["per"][weak_i]["signed"]
print("\nPlain-language readout (blended predictions):")
print("1. Highest per-category MAE: " +
      " and ".join(f"{r['name']} ({r['mae']:.1f} kn, n={r['n']})" for r in worst2) + ".")
n_strong = rep_blended["per"][strong_i]["n"]
n_weak   = rep_blended["per"][weak_i]["n"]
print(f"2. Mean signed error at the strong end ({tci.IMD_NAMES[strong_i]}, n={n_strong}): {ss:+.1f} kn "
      f"({'UNDER' if ss < 0 else 'OVER'}-prediction); at the weak end "
      f"({tci.IMD_NAMES[weak_i]}, n={n_weak}): {sw:+.1f} kn ({'UNDER' if sw < 0 else 'OVER'}-prediction)."
      + ("  [strong-end n is small — indicative only]" if n_strong < 30 else ""))

print("\nIMD thresholds are applied to TCIR best-track wind values as a convenience "
      "mapping. Agencies define and average sustained wind differently, so this is "
      "not an official IMD classification of these storms.")
```

## Cell 10 — 3b: Grad-CAM + randomized-weights sanity check
```python
# ---- Grad-CAM copy: same weights, every ReLU non-inplace ---------------------
model_gc = tci.TCNet(in_channels=1).to(DEVICE)
model_gc.load_state_dict(ckpt["model_state_dict"], strict=True)
n_relu = 0
for m in model_gc.modules():
    if isinstance(m, nn.ReLU):
        m.inplace = False
        n_relu += 1
model_gc.eval()
assert sum(p.numel() for p in model_gc.parameters()) == 389_057
print(f"[OK] Grad-CAM copy: same weights (param count re-verified), "
      f"{n_relu} ReLUs set to inplace=False, eval mode")

# verify the hook target really is the final ReLU with the expected shape
_probe = {}
_h = model_gc.features[11].register_forward_hook(
    lambda m, i, o: _probe.__setitem__("shape", tuple(o.shape)))
with torch.no_grad():
    model_gc(torch.zeros(1, 1, 64, 64, device=DEVICE))
_h.remove()
assert _probe["shape"] == (1, 256, 4, 4), f"features[11] → {_probe['shape']}, expected (1,256,4,4)"
print("[OK] features[11] = final ReLU before GAP; hooked output shape (1,256,4,4)")

# ---- deterministic frame selection -------------------------------------------
true_idx_all = tci.imd_category_indices(y_val)
counts = np.bincount(true_idx_all, minlength=8)
populated = [i for i in range(8) if counts[i] > 0]
top4 = sorted(populated, key=lambda i: (-int(counts[i]), i))[:4]
selection = {}
for c in top4:
    members = np.where(true_idx_all == c)[0]
    med = float(np.median(y_val[members]))
    pos = int(members[int(np.argmin(np.abs(y_val[members] - med)))])
    selection[pos] = f"closest-to-median frame of {tci.IMD_NAMES[c]} (n={int(counts[c])})"
err_abs = np.abs(single_preds - y_val)     # single-pass error: Grad-CAM runs the unrotated forward
worst_order = list(np.argsort(-err_abs, kind="stable"))
# Always include the strongest populated category (the most-populated four are
# usually the weak/moderate ones, and the strong end is where error tends to live).
strong_c = max(populated)
if strong_c not in top4:
    members = np.where(true_idx_all == strong_c)[0]
    med = float(np.median(y_val[members]))
    pos = int(members[int(np.argmin(np.abs(y_val[members] - med)))])
    selection[pos] = (f"closest-to-median frame of {tci.IMD_NAMES[strong_c]} "
                      f"(n={int(counts[strong_c])}) — strongest populated category")
# Exactly 2 largest-error frames (skipping any already selected).
n_worst, k = 0, 0
while n_worst < 2 and k < len(worst_order):
    pos = int(worst_order[k]); k += 1
    if pos not in selection:
        selection[pos] = f"largest |single-pass error| (error rank {k})"
        n_worst += 1
print(f"\n=== 3b — Grad-CAM frame selection (deterministic) ===")
for pos, why in selection.items():
    print(f"  val frame {pos:5d}: {why}  [true {y_val[pos]:.0f} kn, pred {single_preds[pos]:.1f} kn]")

records = []
for pos in selection:
    frame01 = x_val01[pos]
    x_std = preproc.preprocess_unit(frame01)              # ORIGINAL orientation
    cam_disp, cam_raw, pred = tci.grad_cam(model_gc, x_std)
    assert abs(pred - single_preds[pos]) < 1e-3, "Grad-CAM forward disagrees with single-pass prediction"
    records.append({"pos": pos, "base": preproc.crop_unit(frame01), "cam": cam_disp,
                    "raw": cam_raw, "pred": pred, "true": float(y_val[pos]),
                    "cat_i": int(true_idx_all[pos]), "why": selection[pos]})
print(f"[OK] Grad-CAM computed on {len(records)} frames (original orientation; "
      f"forwards match the single-pass predictions)")

# ---- grid figure: IR + overlay per frame, honest caption ----------------------
n = len(records)
fig, axes = plt.subplots(n, 2, figsize=(8.0, 2.75 * n))
axes = np.atleast_2d(axes)
for r, (ax0, ax1) in zip(records, axes):
    ax0.imshow(r["base"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax0.set_title(f"true {r['true']:.0f} kn ({tci.IMD_SHORT[r['cat_i']]}) | "
                  f"pred {r['pred']:.1f} | err {r['pred'] - r['true']:+.1f}", fontsize=9)
    ax1.imshow(r["base"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax1.imshow(r["cam"], cmap="jet", vmin=0, vmax=1, alpha=0.45, interpolation="nearest")
    ax1.set_title("Grad-CAM overlay", fontsize=9)
    for ax in (ax0, ax1):
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Grad-CAM (trained v3) — regions where the output is most sensitive to evidence for\n"
             "HIGHER predicted wind speed. 4×4 CAM upsampled to 64×64: coarse by construction,\n"
             "not proof of eyewall 'understanding'. Base: IR1 crop [0,1], brighter = warmer.",
             fontsize=10)
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
gradcam_path = f"{DRIVE_PHASE3}/gradcam_grid_v3.png"
plt.savefig(gradcam_path, dpi=150)
print(f"Saved {gradcam_path}")
plt.show()

# ---- randomized-weights sanity check (the 2 largest-error frames) -------------
torch.manual_seed(SEED)                        # seeded, freshly initialized TCNet
model_rand = tci.TCNet(in_channels=1).to(DEVICE)   # DEFAULT PyTorch init — a random function
for m in model_rand.modules():
    if isinstance(m, nn.ReLU):
        m.inplace = False
model_rand.eval()
print("\n=== 3b — randomized-weights sanity check ===")
print("random reference: fresh TCNet, default init, torch.manual_seed(42) — "
      "its maps measure how much of the pattern is just input structure")

check_pos = [int(p) for p in np.argsort(-err_abs, kind="stable")[:2]]
SANITY = []
for pos in check_pos:
    x_std = preproc.preprocess_unit(x_val01[pos])
    cam_t, raw_t, pred_t = tci.grad_cam(model_gc, x_std)
    cam_r, raw_r, _      = tci.grad_cam(model_rand, x_std)
    assert abs(pred_t - single_preds[pos]) < 1e-3
    a, b = raw_t.ravel(), raw_r.ravel()   # raw maps; Pearson is invariant to the
    if a.std() == 0 or b.std() == 0:      # per-map [0,1] display rescale (positive affine)
        r, note = float("nan"), "undefined — one map is constant (all zeros)"
    else:
        r, note = float(np.corrcoef(a, b)[0, 1]), ""
    SANITY.append({"pos": pos, "r": r, "note": note, "cam_t": cam_t, "cam_r": cam_r})
    print(f"  val frame {pos}: Pearson r(trained map, random map) = {r:.3f} {note}")

fig, axes = plt.subplots(len(SANITY), 3, figsize=(12.5, 4.1 * len(SANITY)))
axes = np.atleast_2d(axes)
for row, s in enumerate(SANITY):
    pos = s["pos"]
    base = preproc.crop_unit(x_val01[pos])
    ax0, ax1, ax2 = axes[row]
    ax0.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax0.set_title(f"IR1 — true {y_val[pos]:.0f} kn, pred {single_preds[pos]:.1f} "
                  f"(err {single_preds[pos] - y_val[pos]:+.1f})", fontsize=9)
    ax0.set_ylabel(f"val frame {pos}", fontsize=9)
    ax1.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax1.imshow(s["cam_t"], cmap="jet", vmin=0, vmax=1, alpha=0.45, interpolation="nearest")
    ax1.set_title("trained v3 Grad-CAM", fontsize=10)
    ax2.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax2.imshow(s["cam_r"], cmap="jet", vmin=0, vmax=1, alpha=0.45, interpolation="nearest")
    ax2.set_title(f"random-init Grad-CAM  (r = {s['r']:.3f})", fontsize=10)
    for ax in (ax0, ax1, ax2):
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Randomized-weights sanity check — if trained and random maps were highly "
             "correlated, the maps would mostly reflect input structure, not learned behavior",
             fontsize=11)
plt.tight_layout(rect=[0, 0.02, 1, 0.93])
sanity_path = f"{DRIVE_PHASE3}/sanity_check_v3.png"
plt.savefig(sanity_path, dpi=150)
print(f"Saved {sanity_path}")
plt.show()

# ---- observation, data-anchored and worded as an observation ------------------
overlaps = []
for rec in records:
    heat, base = rec["raw"], rec["base"].astype(np.float64)
    if heat.max() <= 0:
        overlaps.append(float("nan")); continue
    hot = heat >= np.percentile(heat, 90)
    cold = base <= np.percentile(base, 20)
    overlaps.append(float((hot & cold).sum()) / max(int(hot.sum()), 1))
valid = [o for o in overlaps if not math.isnan(o)]
mean_ov = float(np.mean(valid)) if valid else float("nan")
rs = [s["r"] for s in SANITY]
mean_r = float(np.nanmean(rs)) if any(not math.isnan(r) for r in rs) else float("nan")

print("\n=== 3b — observation (an observation, not a claim of understanding) ===")
# The wording below follows the number (cutoffs are rough, and with ~6-7 frames the
# estimate is noisy either way) instead of asserting a conclusion in advance.
if math.isnan(mean_ov):
    ov_read = "overlap undefined (constant maps) — inconclusive"
elif mean_ov >= 0.30:
    ov_read = "above the ~20% chance level: the hottest regions lean toward the cold cloud mass"
elif mean_ov >= 0.15:
    ov_read = "close to the ~20% chance level: no clear preference for cold or warm areas"
else:
    ov_read = "below the ~20% chance level: the hottest regions lean toward warmer areas"
print(f"- The hottest 10% of each map overlaps the coldest 20% of the IR crop on average "
      f"{mean_ov:.0%} of the time (chance ~20%), and more than 40% on "
      f"{sum(o > 0.40 for o in valid)}/{len(valid)} frames — {ov_read}.")
if math.isnan(mean_r):
    interp = "correlations undefined (constant maps) — inconclusive"
elif mean_r > 0.7:
    interp = ("HIGH — trained and random maps look alike, so much of the pattern likely "
              "reflects input structure rather than learned behavior")
elif mean_r < 0.3:
    interp = ("low — the trained model's map is largely distinct from a random network's, "
              "consistent with (not proof of) learned behavior")
else:
    interp = "moderate — the maps are partly, but not fully, explained by input structure"
print(f"- Randomized-weights Pearson r = {[f'{r:.3f}' for r in rs]} (mean {mean_r:.3f}) → {interp}.")
print("- At 4×4 CAM resolution upsampled to 64×64 these maps are coarse; they show where "
      "the output is most sensitive, not what a meteorologist would call an eyewall.")
```

## Cell 11 — 3c: Gradio demo (smoke tests first, then launch share=False)
```python
try:
    import gradio as gr
except ImportError:
    import subprocess
    print("installing gradio…")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
    import gradio as gr
print(f"gradio {gr.__version__}")

from matplotlib import colormaps
_GRAY, _JET = colormaps["gray"], colormaps["jet"]

def overlay_rgb(base01_64, cam_disp, alpha=0.45, upscale=4):
    """Display-only composite: grayscale IR + jet Grad-CAM overlay → (H,W,3) uint8."""
    b = np.clip(np.asarray(base01_64, dtype=np.float64), 0.0, 1.0)
    rgb = _GRAY(b)[..., :3]
    heat = _JET(np.clip(cam_disp, 0.0, 1.0))[..., :3]
    img = (((1.0 - alpha) * rgb + alpha * heat) * 255).astype(np.uint8)
    if upscale > 1:
        img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
    return img

# ---- fixed gallery: 20 held-out val frames spanning the intensity range ------
order = np.argsort(y_val, kind="stable")
qpos = np.unique(np.round(np.linspace(0, len(y_val) - 1, 20)).astype(int))
gallery_pos = [int(order[q]) for q in qpos]
LABEL2POS = {}
for p in gallery_pos:
    cat = tci.IMD_NAMES[tci.imd_category_index(y_val[p])]
    LABEL2POS[f"val frame {p} — true {y_val[p]:.0f} kn ({cat})"] = p
gallery_labels = list(LABEL2POS)
gcats = sorted({tci.imd_category_index(y_val[p]) for p in gallery_pos})
print(f"gallery: {len(gallery_labels)} fixed held-out frames, true wind "
      f"{y_val[gallery_pos].min():.0f}–{y_val[gallery_pos].max():.0f} kn, "
      f"categories covered: {[tci.IMD_SHORT[i] for i in gcats]}")

# ---- upload validation: one place, never raises -------------------------------
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

def validate_upload(path):
    """Returns (array_or_None, message). Every check from the spec, in order."""
    try:
        if path is not None and not isinstance(path, str):
            path = getattr(path, "name", None)
        if not path:
            return None, "No file received."
        base = os.path.basename(path)
        if not base.lower().endswith(".npy"):
            return None, ("Rejected: only .npy files are accepted (got "
                          f"'{base}'). The model needs a calibrated 201×201 IR1 "
                          "brightness-temperature array in Kelvin; JPG/PNG screenshots "
                          "have an unknown mapping to Kelvin, so any number they "
                          "produced would look confident and mean nothing.")
        size = os.path.getsize(path)
        if size >= MAX_UPLOAD_BYTES:
            return None, f"Rejected: file is {size/1e6:.1f} MB — the limit is 5 MB."
        try:
            arr = np.load(path, allow_pickle=False)   # NEVER allow_pickle=True on uploads
        except Exception as e:
            return None, ("Rejected: file could not be loaded as a plain .npy array. "
                          "Pickled object arrays are refused on purpose (arbitrary-code "
                          f"execution risk). Loader error: {type(e).__name__}.")
        if arr.ndim != 2:
            return None, f"Rejected: expected a 2-D array, got ndim={arr.ndim}."
        if arr.shape != (201, 201):
            return None, f"Rejected: expected shape (201, 201), got {arr.shape}."
        if arr.dtype.kind not in "fiu":
            return None, f"Rejected: expected a numeric dtype, got {arr.dtype}."
        arrf = arr.astype(np.float64)
        nan_frac = float(np.isnan(arrf).mean())
        if nan_frac > 0.10:
            return None, f"Rejected: {nan_frac*100:.1f}% of pixels are NaN — the limit is 10%."
        valid = arrf[~np.isnan(arrf)]
        if valid.size == 0:
            return None, "Rejected: every pixel is NaN."
        p5, p95 = np.percentile(valid, [5.0, 95.0])
        if not (150.0 <= p5 and p95 <= 330.0):
            return None, (f"Rejected: 5th–95th percentile of non-NaN pixels is "
                          f"[{p5:.1f}, {p95:.1f}] K — outside the 150–330 K IR "
                          "calibration band. This does not look like a calibrated IR1 frame.")
        return arrf, (f"accepted: (201,201) {arr.dtype}, NaN {nan_frac*100:.1f}%, "
                      f"p5–p95 = [{p5:.1f}, {p95:.1f}] K.")
    except Exception as e:
        return None, f"Rejected: unexpected validation problem ({type(e).__name__}: {e})."

DISCLAIMER = ("Estimates the intensity of an existing storm from a single IR frame. "
              "Not a forecast, not trained to predict whether a storm will form. "
              "Validation MAE is about 11 knots overall, but strong storms (above roughly "
              "90 knots) are typically underestimated by 30-40 knots. "
              "Not for operational or safety decisions.")

# ---- the predict function: shared module does ALL preprocessing ---------------
def predict(frame_choice, upload):
    """Gradio handler and smoke-test entry point. NEVER raises on bad input."""
    try:
        pos = None
        if upload is not None:
            path = upload if isinstance(upload, str) else getattr(upload, "name", None)
            arr, msg = validate_upload(path)
            if arr is None:
                return f"**Input rejected**\n\n{msg}", None
            x_std = preproc.preprocess_kelvin(arr)                 # single source of truth
            base = preproc.crop_unit(preproc.kelvin_to_unit(arr))
            src_line = f"Input: uploaded raw 201×201 Kelvin `.npy` — {msg}"
        elif frame_choice:
            pos = LABEL2POS.get(frame_choice)
            if pos is None:
                return "Unknown gallery frame — pick one from the dropdown.", None
            x_std = preproc.preprocess_unit(x_val01[pos])          # single source of truth
            base = preproc.crop_unit(x_val01[pos])
            src_line = f"Input: held-out validation frame {pos}."
        else:
            return ("Pick a held-out frame from the dropdown, or upload a raw 201×201 "
                    "Kelvin `.npy` file."), None

        res = tci.predict_blended(model, x_std)          # same recipe as the repro test
        blended = float(res["blended"][0])
        single  = float(res["single"][0])
        sd      = float(res["spread"][0])

        cam_disp, _, cam_pred = tci.grad_cam(model_gc, x_std)      # original orientation
        assert abs(cam_pred - single) < 1e-3
        img = overlay_rgb(base, cam_disp)

        gt_line = ""
        if pos is not None:
            y_true = float(y_val[pos])
            gt_line = (f"Ground truth: **{y_true:.1f} knots** · error (blended − truth): "
                       f"**{blended - y_true:+.1f} knots**")
        cat = tci.IMD_NAMES[tci.imd_category_index(blended)]
        md = (f"## Blended estimate: **{blended:.1f} knots**\n"
              f"<small>single-pass (original orientation): {single:.1f} knots</small>\n\n"
              f"**IMD category** (of the blended value): {cat}\n\n"
              f"Agreement across rotations (**not a confidence interval**): "
              f"std of the 10 rotated predictions = {sd:.2f} knots\n\n"
              f"{src_line}\n\n{gt_line}\n\n"
              f"*Grad-CAM overlay (original orientation only): regions where the output "
              f"is most sensitive to evidence for HIGHER predicted wind speed.*")
        # Return a Matplotlib figure: in this Colab setup the gr.Image output did not
        # display the overlay, while gr.Plot did. Presentation-layer change only - model,
        # preprocessing, prediction and Grad-CAM are untouched.
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.axis("off")
        plt.close(fig)   # stops Colab also rendering the figure inline
        return md, fig
    except Exception as e:
        return (f"**Unexpected error** ({type(e).__name__}: {e}) — nothing was predicted. "
                "Invalid input should have been rejected earlier; if you see this, "
                "something is wrong with the handler itself."), None

# ---- smoke tests BEFORE launching the UI --------------------------------------
print("=== 3c — demo smoke tests (predict() called directly; UI not launched yet) ===")
SMOKE = []
for lbl in gallery_labels[:3]:
    md, img = predict(lbl, None)
    head = md.splitlines()[0] if md else "(no output)"
    SMOKE.append((lbl, "OK" if img is not None else "NO IMAGE", head))
    print(f"\n--- {lbl} ---\n{md}\n  overlay: {None if img is None else type(img).__name__}")

with open("/tmp/not_really_npy.npy", "wb") as f:
    f.write(b"this file is definitely not a numpy array")
md, img = predict(None, "/tmp/not_really_npy.npy")
SMOKE.append(("invalid upload (garbage bytes in a .npy name)",
              "OK (rejected)" if img is None else "FAIL", md.splitlines()[-1] if md else ""))
print(f"\n--- invalid upload (garbage bytes) ---\n{md}\n  overlay image: {img}")

nan_heavy = np.full((201, 201), 250.0)
nan_heavy.ravel()[::4] = np.nan            # 25% NaN, otherwise plausible values
np.save("/tmp/nan_heavy.npy", nan_heavy)
md, img = predict(None, "/tmp/nan_heavy.npy")
SMOKE.append(("NaN-heavy upload (25% NaN at 250 K)",
              "OK (rejected)" if img is None else "FAIL", md.splitlines()[-1] if md else ""))
print(f"\n--- NaN-heavy upload (25% NaN) ---\n{md}\n  overlay image: {img}")

# ---- end-to-end VALID upload check (the smoke tests above only cover bad files) ---
# A good Kelvin upload must go through validate -> preprocess_kelvin -> model and
# reproduce the gallery path's answer for the same frame.
p_up = gallery_pos[10]
frame_k = (x_val01[p_up].astype(np.float64) * (norm_max - norm_min) + norm_min).astype(np.float32)
np.save("/tmp/good_frame_kelvin.npy", frame_k)
arr_up, msg_up = validate_upload("/tmp/good_frame_kelvin.npy")
assert arr_up is not None, msg_up
b_up  = float(tci.predict_blended(model, preproc.preprocess_kelvin(arr_up))["blended"][0])
b_gal = float(tci.predict_blended(model, preproc.preprocess_unit(x_val01[p_up]))["blended"][0])
print(f"valid-upload path {b_up:.3f} kn | gallery path {b_gal:.3f} kn | diff {abs(b_up - b_gal):.2e}")
assert abs(b_up - b_gal) < 1e-3, "upload path disagrees with the gallery path"
md_up, img_up = predict(None, "/tmp/good_frame_kelvin.npy")
assert img_up is not None, md_up
print("[OK] valid upload accepted, matches gallery path, overlay produced\n")

# ---- the UI --------------------------------------------------------------------
with gr.Blocks(title="TCIR cyclone intensity estimator (v3)") as demo:
    gr.Markdown("### Cyclone intensity estimator — TCIR IR1, CNN v3 (64×64 crop, rotation-augmented)")
    gr.Markdown("Pick a held-out validation frame, **or** upload one raw **201×201 IR1 "
                "brightness-temperature array in Kelvin** as `.npy`. Calibrated temperatures "
                "only: JPG/PNG screenshots and pickled files are rejected — a colorized web "
                "image has an unknown mapping to Kelvin, so any number it produced would "
                "look confident and mean nothing.")
    with gr.Row():
        with gr.Column():
            dd = gr.Dropdown(choices=gallery_labels, value=gallery_labels[0],
                             label="Held-out validation frame (ground truth shown after prediction)")
            up = gr.File(label="Optional upload: raw 201×201 Kelvin .npy (overrides the dropdown)",
                         file_types=[".npy"])
            btn = gr.Button("Estimate intensity", variant="primary")
        with gr.Column():
            out_md = gr.Markdown("Results will appear here after you click **Estimate intensity**.")
            out_img = gr.Plot(label="Grad-CAM overlay — original orientation only")
    gr.Markdown(f"> {DISCLAIMER}")
    btn.click(predict, inputs=[dd, up], outputs=[out_md, out_img])

print("launching with share=False — a share=True tunnel into a runtime with Drive "
      "mounted is only for a deliberate short window, never left running.")
demo.launch(share=False)
```

## Cell 12 — summary block
```python
print("=" * 78)
print("PHASE 3 SUMMARY")
print("=" * 78)
print(f"1. Reproduction test : {'PASS' if (REPRO_SINGLE_OK and REPRO_BLEND_OK) else 'FAIL'}")
print(f"   single-pass MAE {mae_s:.4f} / RMSE {rmse_s:.4f} kn  (ref 11.6101 / 16.6257, tol 0.01)")
print(f"   blended    MAE {mae_b:.4f} / RMSE {rmse_b:.4f} kn  (ref 11.30 / 16.36, tol 0.02)")
print(f"2. Parity test       : {'PASS' if PARITY_OK else 'FAIL'}  "
      f"(max |unit-path − kelvin-path| = {max_diff:.2e}, limit 1e-4)")
print("\n3. Per-category table (blended predictions; full CSV on Drive):")
print(df_cat.round(2).to_string(index=False))
print("   (n<30 rows: from very few storms — indicative only)")
print("\n4. Grad-CAM randomized-weights sanity check:")
for s in SANITY:
    print(f"   val frame {s['pos']}: Pearson r(trained, random) = {s['r']:.3f} {s['note']}")
print("\n5. Demo smoke tests:")
for name, status, head in SMOKE:
    print(f"   [{status:>13s}] {name} → {head}")
print("\n6. Files saved to /content/drive/MyDrive/TCIR/phase3/:")
for fname in ["tc_inference.py", "category_table_v3.csv", "cm_single_v3.png",
              "cm_blended_v3.png", "scatter_pred_vs_true_v3.png",
              "gradcam_grid_v3.png", "sanity_check_v3.png"]:
    p = f"{DRIVE_PHASE3}/{fname}"
    ok = os.path.exists(p)
    print(f"   [{'OK' if ok else 'MISSING':>7s}] {fname}"
          + (f"  ({os.path.getsize(p)/1024:.0f} KB)" if ok else ""))
assert os.path.getmtime(CKPT_PATH) == CKPT_MTIME, \
    "baseline_cnn_v3.pt was modified during Phase 3 — this phase must be read-only!"
print(f"\n[OK] {CKPT_PATH} untouched (mtime unchanged) — no retraining, no overwrite.")
print("[OK] Phase 3 complete.")
```

## Cell 13 — OPTIONAL, demo day only (run AFTER Cell 12 has passed)
```python
# Colab usually can't show a localhost-only Gradio UI inline, so a public link may be
# needed. The app reads nothing from Drive after startup (model, frames and preprocessing
# are all in memory), so unmount Drive first: the public tunnel then has no Drive behind it.
# Cell 12 needs Drive for its file checks, which is why it must run first.
from google.colab import drive
drive.flush_and_unmount()
demo.close()
demo.launch(share=True)
# When finished:  demo.close()
```
