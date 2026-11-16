#!/usr/bin/env python3
"""Download and verify quantized ONNX models for CPU execution."""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Specification of an ONNX model artifact on Hugging Face Hub."""

    role: str
    repo_id: str
    onnx_filename: str
    tokenizer_filename: str = "tokenizer.json"


DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        role="embedding",
        repo_id="onnx-community/Qwen3-Embedding-0.6B-ONNX",
        onnx_filename="onnx/model_int8.onnx",
    ),
    ModelSpec(
        role="reranker",
        repo_id="onnx-community/Qwen3-Reranker-0.6B-ONNX",
        onnx_filename="onnx/model_q4.onnx",
    ),
    ModelSpec(
        role="nli",
        repo_id="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        onnx_filename="onnx/model_quantized.onnx",
    ),
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = ROOT_DIR / "models"


def _read_rss_mb() -> float:
    """Return current process RSS in MiB (Linux /proc/self/status fallback to ru_maxrss)."""
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return kb / 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def download_models(models_dir: Path, roles: set[str] | None = None) -> list[ModelSpec]:
    """Download quantized ONNX weights and tokenizers into ``models/<role>/``."""
    selected: list[ModelSpec] = []
    for spec in DEFAULT_MODELS:
        if roles is not None and spec.role not in roles:
            continue
        target_dir = models_dir / spec.role
        target_dir.mkdir(parents=True, exist_ok=True)
        onnx_target = target_dir / "model.onnx"
        tok_target = target_dir / "tokenizer.json"

        if not onnx_target.exists():
            print(f"[{spec.role}] Downloading {spec.repo_id}/{spec.onnx_filename} ...")
            cached_onnx = hf_hub_download(repo_id=spec.repo_id, filename=spec.onnx_filename)
            shutil.copy2(cached_onnx, onnx_target)
        else:
            print(f"[{spec.role}] Found existing {onnx_target}")

        if not tok_target.exists():
            print(f"[{spec.role}] Downloading {spec.repo_id}/{spec.tokenizer_filename} ...")
            cached_tok = hf_hub_download(repo_id=spec.repo_id, filename=spec.tokenizer_filename)
            shutil.copy2(cached_tok, tok_target)
        else:
            print(f"[{spec.role}] Found existing {tok_target}")

        selected.append(spec)
    return selected


def verify_models(models_dir: Path, specs: list[ModelSpec]) -> None:
    """Inspect ONNX inputs/outputs and run a minimal CPU inference check."""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    rss_start = _read_rss_mb()
    print(f"\nBaseline RSS: {rss_start:.1f} MiB")

    sessions: list[tuple[str, ort.InferenceSession]] = []
    for spec in specs:
        target_dir = models_dir / spec.role
        onnx_path = target_dir / "model.onnx"
        tok_path = target_dir / "tokenizer.json"

        t0 = time.perf_counter()
        tok = Tokenizer.from_file(str(tok_path))
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        load_sec = time.perf_counter() - t0
        sessions.append((spec.role, sess))

        inputs = [(i.name, i.type, i.shape) for i in sess.get_inputs()]
        outputs = [(o.name, o.type, o.shape) for o in sess.get_outputs()]
        print(f"\n[{spec.role}] Loaded in {load_sec:.2f}s | RSS now: {_read_rss_mb():.1f} MiB")
        print(f"  Inputs : {inputs}")
        print(f"  Outputs: {outputs}")

        enc = tok.encode("Проверка запуска модели на CPU", "Тест NLI и эмбеддингов" if spec.role == "nli" else None)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        input_names = {i.name for i in sess.get_inputs()}

        feed: dict[str, np.ndarray] = {}
        if "input_ids" in input_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in input_names:
            feed["attention_mask"] = attention_mask
        if "position_ids" in input_names:
            pos = np.cumsum(attention_mask, axis=1) - 1
            feed["position_ids"] = np.where(attention_mask > 0, pos, 0).astype(np.int64)
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)

        missing = [i.name for i in sess.get_inputs() if i.name not in feed]
        if missing:
            print(f"  WARNING: extra required/optional inputs not in basic feed: {missing[:6]} (total={len(missing)})")
            # For causal LMs with KV cache inputs (past_key_values.*), synthesize empty cache
            for inp in sess.get_inputs():
                if inp.name not in feed and inp.name.startswith("past_key_values"):
                    shape = [1 if not isinstance(d, int) or d <= 0 else d for d in inp.shape]
                    # sequence length dimension (usually index 2) should be 0 for initial step
                    if len(shape) == 4:
                        shape[2] = 0
                    dtype = np.float16 if "float16" in inp.type else np.float32
                    feed[inp.name] = np.zeros(shape, dtype=dtype)

        t1 = time.perf_counter()
        out = sess.run(None, feed)
        infer_ms = (time.perf_counter() - t1) * 1000.0
        print(f"  Inference OK in {infer_ms:.1f} ms | out[0].shape={np.asarray(out[0]).shape} | RSS: {_read_rss_mb():.1f} MiB")

    print(f"\nAll {len(sessions)} sessions resident RSS: {_read_rss_mb():.1f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup quantized ONNX models for ml-pipeline-py.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--only", nargs="*", help="Subset of roles: embedding, reranker, nli")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX sessions and measure RAM/latency")
    parser.add_argument("--skip-download", action="store_true", help="Only run verification on existing files")
    args = parser.parse_args()

    roles = set(args.only) if args.only else None
    if args.skip_download:
        specs = [s for s in DEFAULT_MODELS if roles is None or s.role in roles]
    else:
        specs = download_models(args.models_dir, roles=roles)

    if args.verify:
        verify_models(args.models_dir, specs)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
