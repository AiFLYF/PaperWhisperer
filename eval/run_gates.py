#!/usr/bin/env python3
"""Release-gate CLI: run the plan's section 7 gates and fail the build on a real one.

Two modes, and the difference between them is the point of this file
-----------------------------------------------------------------------

**CI mode** (the default) answers *"did anything regress?"*. It exits non-zero
only when a gate actually **failed**. A gate that could not run -- because
``eval/corpus/`` is gitignored and absent on a fresh clone, or because
``PAPERWHISPERER_TEST_PG_DSN`` is unset, or because the label set has not been
built yet -- exits **zero** and says so loudly. That asymmetry is deliberate: a
gate that fails a pull request for want of a download teaches everyone that
gate output is noise, and they will be right.

**Release mode** (``--release``) answers *"may this ship?"* and refuses to let
an *undecided* gate through. Two of the seven gates are currently blocked on a
labelling job, so ``--release`` today exits non-zero and names them. That is
the correct answer for a public beta, and it is why the two modes are separate:
the build rule and the release rule are different questions about the same data.

The four statuses, and why there are four
-----------------------------------------
``PASS`` / ``FAIL`` / ``SKIP`` / ``BLOCKED``. ``SKIP`` means "this environment
lacks a fixture" -- materialise it and the gate runs. ``BLOCKED`` means "this
measurement has no ground truth" -- only a labelling job fixes it, and no
amount of downloading will. Neither is green. The summary prints them as
separate counts, and ``--release`` treats both as blockers, so "the gate did not
run" can never be read as "the gate passed".

Usage
-----
::

    python -m eval.run_gates                        # all gates, CI rules
    python -m eval.run_gates --only silent_truncation
    python -m eval.run_gates --list                 # thresholds, run nothing
    python -m eval.run_gates --release              # refuse to ship on undecided
    python -m eval.run_gates --limit 3              # quick corpus smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

# ``eval/`` has no ``__init__.py``, so bootstrapping the repo root is what lets
# ``python -m eval.run_gates`` import both ``eval.gates`` and ``paperwhisperer``.
# Mirrors the same block at the top of ``eval/run_parser_eval.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.gates import (  # noqa: E402
    ALL_GATES,
    DEFAULT_LABELS,
    DEFAULT_MANIFEST,
    DEFAULT_RECALL_K,
    DEFAULT_REPORT_DIR,
    GATES,
    GATES_BY_KEY,
    STATUS_LABELS_ZH,
    GateContext,
    ReleaseNotReady,
    assert_release_ready,
    corpus_stats,
    render_text,
    run_gates,
    write_report,
)

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_BAD_USAGE = 2
EXIT_NOT_READY = 3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_gates.py",
        description=(
            "执行 docs/DEVELOPMENT_PLAN.md 第 7 节的发布门禁。"
            "默认只在一门真正失败时返回非零；--release 模式下未判定的门禁同样阻断。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m eval.run_gates --list\n"
            "  python -m eval.run_gates --only direct_quote_alignment,silent_truncation\n"
            "  python -m eval.run_gates --limit 3            # 语料冒烟\n"
            "  python -m eval.run_gates --release            # 发布前硬门禁\n"
        ),
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="manifest.yaml 路径")
    parser.add_argument(
        "--corpus-root", default=None, help="覆盖 manifest 中的 corpus_root",
    )
    parser.add_argument(
        "--parsers", default="legacy",
        help=(
            "语料门禁使用的解析器，逗号分隔，按顺序兜底。默认 legacy："
            "无额外依赖、52 份约 80s。生产默认是 docling（慢约 20 倍），"
            "实测生产路由请传 --parsers docling,legacy"
        ),
    )
    parser.add_argument(
        "--labels", default=str(DEFAULT_LABELS),
        help="标注集路径（引用精度与 Recall@K 需要；默认不存在，两门为 BLOCKED）",
    )
    parser.add_argument(
        "--only", default="",
        help=f"逗号分隔的门禁名，留空表示全部：{','.join(ALL_GATES)}",
    )
    parser.add_argument(
        "--recall-k", type=int, default=DEFAULT_RECALL_K,
        help=f"Recall@K 的 K（默认 {DEFAULT_RECALL_K}）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只跑清单排序后的前 N 份文档（按 (类别, 文件) 排序，保证每次选中同一批）",
    )
    parser.add_argument(
        "--skip-subprocess", action="store_true",
        help="不跑以子进程方式执行的测试门禁（零出网、任务恢复）",
    )
    parser.add_argument(
        "--recovery-tiers", default="personal,hosted",
        help="任务恢复门禁要跑的档位，逗号分隔：personal,hosted",
    )
    parser.add_argument("--out", default=str(DEFAULT_REPORT_DIR), help="报告输出目录")
    parser.add_argument(
        "--release", action="store_true",
        help="发布模式：任何未判定（跳过或阻塞）的门禁都阻断发布",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_gates",
        help="只打印门禁定义与阈值，不执行任何门禁",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出摘要")
    parser.add_argument("--quiet", action="store_true", help="只输出关键行")
    return parser


def _print_gate_list() -> None:
    """The threshold table, without running anything.

    Printing this on a machine with no corpus is the point: the thresholds are
    properties of the *definition*, not of the environment, so a reviewer can
    check "is this gate actually enforcing 100%?" from a clean clone.
    """
    print("Release gate definitions / 发布门禁定义")
    print()
    print("| Gate | 要求 | 阈值 | 需要什么 |")
    print("|---|---|---|---|")
    for spec in GATES:
        needs = spec.needs.replace("|", "\\|")
        print(
            f"| `{spec.key}` | {spec.requirement} | `{spec.threshold_text}` | {needs} |"
        )
    print()
    blocked = [spec.key for spec in GATES if spec.blocked]
    if blocked:
        print(
            f"阻塞门禁 blocked on a labelling job: {'、'.join(blocked)}。"
            "它们的阈值已经写死，但没有标注集就没有可计算的真值，"
            "因此不产出任何数字——见 docs/RELEASE_GATES.md。"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    :return: ``0`` no gate failed; ``1`` a gate failed; ``2`` bad usage;
        ``3`` ``--release`` was given and a gate was failed or undecided.
    """
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    if args.list_gates:
        _print_gate_list()
        return EXIT_OK

    selected = [name.strip() for name in str(args.only).split(",") if name.strip()]
    unknown = [name for name in selected if name not in GATES_BY_KEY]
    if unknown:
        print(f"error: 未知的门禁名 {unknown}；可选：{','.join(ALL_GATES)}", file=sys.stderr)
        return EXIT_BAD_USAGE

    tiers = tuple(
        item.strip() for item in str(args.recovery_tiers).split(",") if item.strip()
    )
    parsers = tuple(
        item.strip() for item in str(args.parsers).split(",") if item.strip()
    )
    context = GateContext(
        manifest_path=Path(args.manifest),
        corpus_root=Path(args.corpus_root) if args.corpus_root else None,
        labels_path=Path(args.labels),
        recall_k=int(args.recall_k),
        limit=args.limit,
        run_tests=not args.skip_subprocess,
        recovery_tiers=tiers,
        parsers=parsers or ("legacy",),
    )

    run = run_gates(context, only=selected)

    written = write_report(run, Path(args.out))

    if args.json:
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_text(run))
        if not args.quiet:
            print("已写入 / written:")
            for path in written:
                print(f"  {path}")
            stats = corpus_stats(context)
            if stats.get("available") and not stats.get("at_target"):
                print(
                    f"注 note: 语料 {stats['present']} 份，"
                    f"计划的公开发布目标为 {stats['target_min']}-{stats['target_max']} 份"
                    "（T4.3）。语料扩充是项目所有者的数据收集任务，"
                    "不作为门禁失败，但两份标注来源依赖它。"
                )

    if args.release:
        try:
            assert_release_ready(run)
        except ReleaseNotReady as exc:
            print(str(exc), file=sys.stderr)
            print(
                "发布模式 --release 下，未判定的门禁与失败的门禁同样阻断发布。",
                file=sys.stderr,
            )
            return EXIT_NOT_READY
        print("发布门禁全部通过 / all release gates passed.")

    if run.failed:
        names = "、".join(
            f"{result.key}({STATUS_LABELS_ZH[result.status]})" for result in run.failed
        )
        print(f"以下门禁失败 / the following gates failed: {names}", file=sys.stderr)
        return EXIT_GATE_FAILED

    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())