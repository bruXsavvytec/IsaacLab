#!/usr/bin/env python
"""V-JEPA 2 inference: video -> embeddings (and optional action classification).

V-JEPA 2 (Meta FAIR) is a self-supervised video encoder + a latent predictor world
model. This script loads a checkpoint via HuggingFace transformers and runs two modes:

  features  -> encoder last_hidden_state (per-patch embeddings) + a pooled clip vector,
               plus the predictor output. This is the general-purpose representation.
  classify  -> uses a fine-tuned classification head (e.g. Something-Something-V2) to
               predict an action label from the clip.

Video decoding uses PyAV (`av`), NOT torchcodec: torchcodec is installed in isaac-env
but fails to import (no system ffmpeg libav* libs). PyAV ships its own ffmpeg and works.

Examples
--------
    # Feature extraction from a local clip
    isaac-env/bin/python IsaacLab/scripts/vla/jepa_infer.py \
        --video /path/to/clip.mp4

    # Feature extraction from a URL (PyAV opens http directly)
    isaac-env/bin/python IsaacLab/scripts/vla/jepa_infer.py \
        --video https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4

    # No video? synthesizes a dummy clip so you can smoke-test the model end to end
    isaac-env/bin/python IsaacLab/scripts/vla/jepa_infer.py

    # Action classification with the SSv2-finetuned head
    isaac-env/bin/python IsaacLab/scripts/vla/jepa_infer.py \
        --mode classify --model facebook/vjepa2-vitl-fpc16-256-ssv2 --video clip.mp4
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


# Default encoder checkpoint (feature extraction). For --mode classify, pass an
# *-ssv2 (or other finetuned) checkpoint via --model.
DEFAULT_ENCODER = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_CLASSIFIER = "facebook/vjepa2-vitl-fpc16-256-ssv2"


def sample_frames(video_path: str, num_frames: int) -> np.ndarray:
    """Decode `num_frames` RGB frames, evenly spaced across the whole clip.

    Returns a uint8 array of shape (T, H, W, C). Uses PyAV so it works without
    torchcodec / system ffmpeg libs, and accepts both local paths and URLs.
    """
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        # total_frames is often available from metadata; fall back to a full decode.
        total = stream.frames or 0
        if total > 0:
            want = set(np.linspace(0, total - 1, num=num_frames, dtype=int).tolist())
            frames, idx = [], 0
            for frame in container.decode(stream):
                if idx in want:
                    frames.append(frame.to_ndarray(format="rgb24"))
                idx += 1
                if len(frames) == num_frames:
                    break
        else:
            # Unknown length: decode all, then subsample.
            all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
            if not all_frames:
                raise RuntimeError(f"No frames decoded from {video_path!r}")
            sel = np.linspace(0, len(all_frames) - 1, num=num_frames, dtype=int)
            frames = [all_frames[i] for i in sel]

    # If the clip was shorter than requested, pad by repeating the last frame.
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.stack(frames[:num_frames], axis=0)  # (T, H, W, C) uint8


def dummy_frames(num_frames: int, size: int = 256) -> np.ndarray:
    """A moving-gradient synthetic clip for smoke-testing without a real video."""
    t = np.linspace(0, 1, num_frames)[:, None, None, None]
    base = np.linspace(0, 255, size, dtype=np.float32)[None, :, None, None]
    vid = (base + 255 * t) % 256  # drifting brightness over time -> some "motion"
    vid = np.broadcast_to(vid, (num_frames, size, size, 3))
    return vid.astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(description="V-JEPA 2 video inference")
    p.add_argument("--video", default=None,
                   help="Path or URL to a video. If omitted, a dummy clip is used.")
    p.add_argument("--mode", choices=["features", "classify"], default="features")
    p.add_argument("--model", default=None,
                   help=f"HF checkpoint. Default: {DEFAULT_ENCODER} (features) "
                        f"or {DEFAULT_CLASSIFIER} (classify).")
    p.add_argument("--frames", type=int, default=None,
                   help="Number of frames to sample (default: model's frames_per_clip).")
    p.add_argument("--topk", type=int, default=5, help="Top-k labels in classify mode.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    from transformers import AutoVideoProcessor

    if args.mode == "classify":
        from transformers import AutoModelForVideoClassification as ModelCls
        repo = args.model or DEFAULT_CLASSIFIER
    else:
        from transformers import AutoModel as ModelCls
        repo = args.model or DEFAULT_ENCODER

    print(f"[jepa] device={device} dtype={dtype} mode={args.mode}")
    print(f"[jepa] loading {repo} ...")
    processor = AutoVideoProcessor.from_pretrained(repo)
    model = ModelCls.from_pretrained(repo, torch_dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()

    # How many frames to feed. The processor/model knows its pretraining clip length.
    n = args.frames or getattr(model.config, "frames_per_clip", 64)

    if args.video:
        print(f"[jepa] decoding {n} frames from {args.video}")
        video = sample_frames(args.video, n)
    else:
        print(f"[jepa] no --video given; using a {n}-frame synthetic clip")
        video = dummy_frames(n)
    print(f"[jepa] frames: {video.shape} dtype={video.dtype}")

    # Processor handles resize/crop/normalize and returns pixel_values_videos.
    inputs = processor(list(video), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if args.mode == "classify":
        logits = outputs.logits.float()
        probs = torch.softmax(logits, dim=-1)
        k = min(args.topk, logits.shape[-1])
        top = probs.topk(k, dim=-1)
        print(f"\n[jepa] top-{k} predictions:")
        for prob, idx in zip(top.values[0], top.indices[0]):
            label = model.config.id2label.get(idx.item(), str(idx.item()))
            print(f"  {prob.item():6.3f}  {label}")
    else:
        # Encoder per-patch embeddings: (B, num_patches, hidden_size).
        enc = outputs.last_hidden_state.float()
        # A single clip vector via mean-pool over patches — handy for retrieval/probing.
        clip_vec = enc.mean(dim=1)
        print(f"\n[jepa] encoder last_hidden_state: {tuple(enc.shape)}")
        print(f"[jepa] pooled clip embedding:     {tuple(clip_vec.shape)} "
              f"(norm={clip_vec.norm(dim=-1).item():.3f})")
        pred = getattr(outputs, "predictor_output", None)
        if pred is not None and getattr(pred, "last_hidden_state", None) is not None:
            print(f"[jepa] predictor last_hidden_state: "
                  f"{tuple(pred.last_hidden_state.shape)}")

    print("\n[jepa] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
