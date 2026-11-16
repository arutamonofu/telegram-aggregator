#!/usr/bin/env python3
"""Агрегатор отчетов качества и проверки готовности к переходу Stage 1 -> Stage 2 (MVP).

Проверяет инварианты разделов 8.1.4 (Stage 1 Quality Gates) и 9.2 (5-шаговая
калибровка + сквозной сценарий gotd Layer 225) `docs/SPEC.md` по всем
трем контурам: Go (`services/ingestion-go`), Python (`services/ml-pipeline-py`)
и TypeScript (`apps/web-client`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
ML_SERVICE_DIR = ROOT_DIR / "services" / "ml-pipeline-py"
GO_SERVICE_DIR = ROOT_DIR / "services" / "ingestion-go"
WEB_CLIENT_DIR = ROOT_DIR / "apps" / "web-client"


@dataclass(slots=True)
class SuiteResult:
    name: str
    command: list[str]
    returncode: int
    passed: bool
    summary: str
    output: str


def _resolve_go_binary_and_env() -> tuple[str | None, dict[str, str]]:
    env = os.environ.copy()
    candidates = [
        shutil.which("go"),
        "/tmp/gopath/pkg/mod/golang.org/toolchain@v0.0.1-go1.26.11.linux-amd64/bin/go",
        "/tmp/go-sdk/go/bin/go",
        "/usr/local/go/bin/go",
    ]
    go_bin = next((c for c in candidates if c and Path(c).is_file()), None)
    if go_bin:
        go_dir = str(Path(go_bin).parent)
        env["PATH"] = f"{go_dir}:{env.get('PATH', '')}"
        if "/tmp/gopath" in go_bin or Path("/tmp/gopath").exists():
            env.setdefault("GOPATH", "/tmp/gopath")
            env.setdefault("GOTOOLCHAIN", "local")
        if Path("/tmp/gocache").exists() or "GOCACHE" not in env:
            env.setdefault("GOCACHE", "/tmp/gocache")
    return go_bin, env


def _run_subprocess(name: str, cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> SuiteResult:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    return SuiteResult(
        name=name,
        command=cmd,
        returncode=proc.returncode,
        passed=proc.returncode == 0,
        summary="",
        output=combined,
    )


def _run_go_suite() -> SuiteResult:
    go_bin, go_env = _resolve_go_binary_and_env()
    if not go_bin:
        return SuiteResult(
            name="Go Ingestion (services/ingestion-go)",
            command=["go", "test", "./..."],
            returncode=127,
            passed=False,
            summary="go binary not found",
            output="go compiler binary not found in PATH or known toolchain paths",
        )
    res = _run_subprocess(
        "Go Ingestion (services/ingestion-go)",
        [go_bin, "test", "-C", str(GO_SERVICE_DIR), "-count=1", "./..."],
        cwd=ROOT_DIR,
        env=go_env,
    )
    ok_pkgs = len(re.findall(r"^ok\s+", res.output, flags=re.MULTILINE))
    # Целевая проверка конкретного Quality Gate 8.1.4 #1 (72ч x 300 каналов)
    gate_res = _run_subprocess(
        "Go Ingestion Gate #1",
        [
            go_bin,
            "test",
            "-C",
            str(GO_SERVICE_DIR),
            "-v",
            "-count=1",
            "./internal/mtproto",
            "-run",
            "TestStage1QualityGate_72h300ChannelsZeroPtsDesyncAndFloodWait",
        ],
        cwd=ROOT_DIR,
        env=go_env,
    )
    gate_ok = gate_res.passed and "--- PASS: TestStage1QualityGate_72h300ChannelsZeroPtsDesyncAndFloodWait" in gate_res.output
    res.passed = res.passed and gate_ok
    res.summary = f"{ok_pkgs} Go packages ok; 72h x 300 channels gate={'PASS' if gate_ok else 'FAIL'}"
    if not gate_ok:
        res.output = res.output + "\n--- Gate #1 ---\n" + gate_res.output
    return res


def _run_python_suite() -> SuiteResult:
    py_exe = sys.executable
    res = _run_subprocess(
        "Python ML Pipeline (services/ml-pipeline-py)",
        [py_exe, "-m", "pytest", str(ML_SERVICE_DIR / "tests"), "-q"],
        cwd=ROOT_DIR,
    )
    m = re.search(r"(\d+)\s+passed", res.output)
    passed_count = m.group(1) if m else "?"
    res.summary = f"{passed_count} pytest cases passed"
    return res


def _run_typescript_suite() -> SuiteResult:
    bun_bin = shutil.which("bun")
    if not bun_bin:
        return SuiteResult(
            name="TypeScript Web Client (apps/web-client)",
            command=["bun", "test", "--cwd", str(WEB_CLIENT_DIR)],
            returncode=127,
            passed=False,
            summary="bun binary not found",
            output="bun runtime not found in PATH",
        )
    res = _run_subprocess(
        "TypeScript Web Client (apps/web-client)",
        [bun_bin, "test", "--cwd", str(WEB_CLIENT_DIR)],
        cwd=ROOT_DIR,
    )
    m_pass = re.search(r"^\s*(\d+)\s+pass", res.output, flags=re.MULTILINE)
    m_fail = re.search(r"^\s*(\d+)\s+fail", res.output, flags=re.MULTILINE)
    pass_cnt = m_pass.group(1) if m_pass else "?"
    fail_cnt = m_fail.group(1) if m_fail else "0"
    res.passed = res.passed and fail_cnt == "0"
    res.summary = f"{pass_cnt} passed, {fail_cnt} failed"
    return res


def _check_zero_dangerously_set_inner_html() -> tuple[bool, int]:
    src_root = WEB_CLIENT_DIR / "src"
    scanned = 0
    for path in src_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".ts", ".tsx", ".js", ".jsx"}:
            continue
        if "__tests__" in path.parts or ".test." in path.name:
            continue
        scanned += 1
        if "dangerouslySetInnerHTML" in path.read_text(encoding="utf-8"):
            return False, scanned
    return scanned > 0, scanned


def build_quality_gate_report() -> dict[str, Any]:
    """Запускает все тестовые контуры и собирает единый отчет готовности MVP."""
    go_suite = _run_go_suite()
    py_suite = _run_python_suite()
    ts_suite = _run_typescript_suite()
    no_raw_html, ts_files_scanned = _check_zero_dangerously_set_inner_html()

    if str(ML_SERVICE_DIR) not in sys.path:
        sys.path.insert(0, str(ML_SERVICE_DIR))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from app.calibration_qa import (  # noqa: PLC0415
            evaluate_stage1_quality_gates,
            run_spec_9_2_e2e_scenario,
        )

        e2e = run_spec_9_2_e2e_scenario()
        ml_gates = evaluate_stage1_quality_gates(spec_9_2_result=e2e)

    e2e_clean = {k: v for k, v in e2e.items() if k != "pipeline_artifacts"}

    # Проверка сквозных инвариантов SPEC 9.2
    spec_9_2_checks = {
        "p_ad_filtered_l1_over_0_80": (
            e2e_clean["ad_verdict"]["is_ad"] is True
            and e2e_clean["ad_verdict"]["l1_score"] > 0.80
            and 9200 not in e2e_clean["retrieval"]["reranked_post_ids"]
        ),
        "p_target_and_3_rewrites_clustered_with_origin": (
            e2e_clean["clustering"]["member_post_ids"] == [9201, 9202, 9203, 9204]
            and e2e_clean["clustering"]["origin_post_id"] == 9201
            and e2e_clean["clustering"]["neg_cluster_id"] != e2e_clean["clustering"]["target_cluster_id"]
        ),
        "bm25_exact_boost_1_6_and_cache_tokens": (
            {"225", "gotd"}.issubset(set(e2e_clean["entities_and_cache"]["extracted_all_canonical"]))
            and e2e_clean["entities_and_cache"]["bm25_weight"] == 1.6
        ),
        "selectivity_under_0_5_pct_and_dense_code_over_0_55": (
            e2e_clean["retrieval"]["selectivity_under_0_5_pct"] is True
            and e2e_clean["retrieval"]["dense_code_cosine"] > 0.55
            and e2e_clean["retrieval"]["group_by"] == "post_id"
            and e2e_clean["retrieval"]["top1_post_id"] == 9201
        ),
        "nli_gate_and_utf16_emoji_shift": (
            e2e_clean["rag_and_nli"]["citation_post_id"] == 9201
            and e2e_clean["rag_and_nli"]["utf16_emoji_shift"] == 1
            and e2e_clean["rag_and_nli"]["sw_alignment_score"] >= 0.85
            and e2e_clean["rag_and_nli"]["nli_entailment_score"] >= 0.75
            and "updates.channelDifferenceTooLong" in e2e_clean["rag_and_nli"]["utf16_sliced_text"]
        ),
    }

    quality_gates_8_1_4 = [
        {
            "id": "8.1.4-1",
            "name": "MTProto Ingestion Stability (72h x 300 channels)",
            "target": "0 pts desync & 0 unhandled FLOOD_WAIT",
            "observed": go_suite.summary,
            "passed": go_suite.passed,
        },
        {
            "id": "8.1.4-2",
            "name": "Spam Filter Cascade (L0 -> L1 -> L2)",
            "target": "Recall >= 96%, FPR <= 1.5%, L0+L1 CPU >= 82%",
            "observed": (
                f"Recall={ml_gates['spam_cascade']['recall']:.1%}, "
                f"FPR={ml_gates['spam_cascade']['fpr']:.1%}, "
                f"CPU={ml_gates['spam_cascade']['l0_l1_cpu_share']:.1%}"
            ),
            "passed": ml_gates["spam_cascade"]["passed"],
        },
        {
            "id": "8.1.4-3",
            "name": "Single-Pass Clustering & OriginScore",
            "target": "ARI >= 0.82, Homogeneity >= 0.88, Origin Top-1 >= 90%",
            "observed": (
                f"ARI={ml_gates['clustering_and_origin']['ari']:.3f}, "
                f"Homog={ml_gates['clustering_and_origin']['homogeneity']:.3f}, "
                f"OriginTop1={ml_gates['clustering_and_origin']['origin_top1_accuracy']:.1%}"
            ),
            "passed": ml_gates["clustering_and_origin"]["passed"],
        },
        {
            "id": "8.1.4-4",
            "name": "Hybrid Search + Reranking (Weighted RRF)",
            "target": "NDCG@10 >= 0.84, MRR@5 >= 0.88, Recall@20 >= 0.95",
            "observed": (
                f"NDCG@10={ml_gates['hybrid_search_and_rerank']['ndcg_at_10']:.3f}, "
                f"MRR@5={ml_gates['hybrid_search_and_rerank']['mrr_at_5']:.3f}, "
                f"Recall@20={ml_gates['hybrid_search_and_rerank']['recall_at_20']:.3f}"
            ),
            "passed": ml_gates["hybrid_search_and_rerank"]["passed"],
        },
        {
            "id": "8.1.4-5",
            "name": "Glass-box NLI & Smith-Waterman UTF-16",
            "target": "CitationPrec=100%, Leakage<=0.5%, UTF16Exact=100%",
            "observed": (
                f"Prec={ml_gates['nli_and_smith_waterman']['citation_precision']:.1%}, "
                f"Leakage={ml_gates['nli_and_smith_waterman']['hallucination_leakage_rate']:.2%}, "
                f"UTF16={ml_gates['nli_and_smith_waterman']['utf16_exact_match']:.1%}"
            ),
            "passed": ml_gates["nli_and_smith_waterman"]["passed"],
        },
        {
            "id": "8.1.4-6",
            "name": "Two-Level Semantic Cache Isolation",
            "target": "0% false hits on numeric/entity conflicts (225 vs 224)",
            "observed": f"false_cache_hits={ml_gates['semantic_cache_isolation']['false_cache_hits']}",
            "passed": ml_gates["semantic_cache_isolation"]["passed"],
        },
        {
            "id": "8.1.4-7",
            "name": "Web Client AST Security & Zero UTF-16 Shift",
            "target": "0 dangerouslySetInnerHTML & 0 DOM citation shift",
            "observed": f"{ts_suite.summary}; scanned_src_files={ts_files_scanned}, raw_html={not no_raw_html}",
            "passed": ts_suite.passed and no_raw_html,
        },
    ]

    all_passed = (
        go_suite.passed
        and py_suite.passed
        and ts_suite.passed
        and all(g["passed"] for g in quality_gates_8_1_4)
        and all(spec_9_2_checks.values())
    )

    return {
        "mvp_ready": all_passed,
        "suites": [
            {"name": s.name, "passed": s.passed, "returncode": s.returncode, "summary": s.summary}
            for s in (go_suite, py_suite, ts_suite)
        ],
        "quality_gates_8_1_4": quality_gates_8_1_4,
        "spec_9_2_e2e_checks": spec_9_2_checks,
        "spec_9_2_telemetry": e2e_clean,
        "failed_outputs": {
            s.name: s.output for s in (go_suite, py_suite, ts_suite) if not s.passed
        },
    }


def _print_human_report(report: dict[str, Any]) -> None:
    print("=" * 84)
    print("Stage 1 -> Stage 2 (MVP) Readiness Report (docs/SPEC.md Sections 8.1.4 & 9.2)")
    print("=" * 84)
    print("\n[1] Test Suites Aggregation")
    for s in report["suites"]:
        mark = "PASS" if s["passed"] else "FAIL"
        print(f"  - [{mark}] {s['name']}: {s['summary']}")

    print("\n[2] SPEC 8.1.4 Quality Gates")
    for gate in report["quality_gates_8_1_4"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        print(f"  - [{mark}] {gate['id']} {gate['name']}")
        print(f"      Target:   {gate['target']}")
        print(f"      Observed: {gate['observed']}")

    print("\n[3] SPEC 9.2 E2E Walkthrough ('gotd Layer 225')")
    for key, ok in report["spec_9_2_e2e_checks"].items():
        mark = "PASS" if ok else "FAIL"
        print(f"  - [{mark}] {key}")
    tel = report["spec_9_2_telemetry"]
    print(
        f"      Telemetry: top1_post_id={tel['retrieval']['top1_post_id']}, "
        f"bm25_w={tel['entities_and_cache']['bm25_weight']}, "
        f"dense_code_cos={tel['retrieval']['dense_code_cosine']}, "
        f"sw={tel['rag_and_nli']['sw_alignment_score']}, "
        f"nli={tel['rag_and_nli']['nli_entailment_score']}, "
        f"utf16_range={tel['rag_and_nli']['utf16_range']}"
    )

    if report["failed_outputs"]:
        print("\n[!] Failed Suite Details")
        for name, out in report["failed_outputs"].items():
            print(f"--- {name} ---\n{out}")

    verdict = "READY FOR STAGE 2 (MVP)" if report["mvp_ready"] else "BLOCKED (QUALITY GATES FAILED)"
    print(f"\nOVERALL VERDICT: {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SPEC 8.1.4 & 9.2 MVP Quality Gates.")
    parser.add_argument("--json", action="store_true", help="Output JSON report instead of text table.")
    args = parser.parse_args()

    report = build_quality_gate_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human_report(report)
    return 0 if report["mvp_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
