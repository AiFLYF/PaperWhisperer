#!/usr/bin/env python3
"""T4.2 security runner: the injection corpus, and a stdlib SBOM generator.

Two subcommands, one entry point
--------------------------------

``injection``
    Run the document prompt-injection corpus from :mod:`eval.injection.harness`
    and print a report. Offline by default — the structural half of the corpus
    needs no endpoint — and ``--live`` to additionally spend real completions
    against a configured OpenAI-compatible endpoint.

``sbom``
    Emit a CycloneDX 1.5 SBOM from the dependencies ``pyproject.toml``
    *declares*.

**Why an SBOM generator lives in a file called ``run_injection_eval.py``:** it
does not, ideally, and the honest reason it is here is that this task was
granted exactly four paths to write and ``tools/run_injection_eval.py`` is the
only one of them that is a tool. Both subcommands are T4.2 release-gate
artefacts, so one command that produces both security reports is defensible; a
second file would have been better. Splitting ``sbom`` into
``tools/generate_sbom.py`` is a pure move of one function and one ``main``
branch, and it is the right next step for whoever next touches ``tools/``.

What this SBOM is, and what it is not
-------------------------------------

``cyclonedx-py`` is the tool for this job and it is not installed, and nothing
is installed here. So this is a **declaration** SBOM, not an **environment**
SBOM, and the difference is most of what follows:

* It lists what ``pyproject.toml`` *asks for* — the base dependencies and each
  optional extra — as CycloneDX components with their declared specifier.
* **``component.version`` is a ``>=`` floor, not a resolved version.** A
  consumer that reads it as "the version in use" will be wrong, and a
  vulnerability scanner matching on it will produce arbitrary results. The
  declared specifier is carried in an explicit ``properties`` entry so the
  distinction is not left to inference, and the ``uv.lock`` resolution for the
  same distribution sits beside it as ``pw:locked_version``.
* **The installed environment is reported separately, and separately is the
  operative word.** ``importlib.metadata`` knows what is on *this* machine,
  which in CI is not the environment the wheel ships into and specifically is
  not one with the ``parse`` extra installed. Those versions are emitted under
  an ``environment`` property and never as ``component.version``.
* **No transitive closure.** ``openai`` pulls in ``httpx``, ``pydantic``,
  ``anyio`` and more; none of them appear. An SBOM with no transitive
  dependencies describes roughly a tenth of a Python environment.
* **No licences and no integrity hashes.** ``uv.lock`` carries a ``sha256`` for
  every artifact it resolved and ``pyproject.toml`` declares no licences, and
  neither is copied in: the hash that matters is of the bytes a deployment
  actually installed, and this document describes declared dependencies rather
  than an installed set. A lock's hash quoted here would read as "verified"
  while verifying nothing.
* **No vulnerability data at all.** CVE correlation is a scanner's job, and it
  is the job this generator is least qualified for, precisely because the
  versions it would match against are floors.

A real ``cyclonedx-py`` run in a pinned environment would add: the full
transitive graph from the lock, hashes read off the installed distributions,
licence metadata, and (with a scanner) matched advisories. The gap is
enumerated at the end of every run so it is stated where it is read.

The lockfile
------------

The project has one: ``uv.lock``, tracked at the repository root, pinning every
package it resolved. This tool reads it and never writes it -- refreshing a lock
is ``uv lock``, and doing that as a side effect of running an evaluation would
repin ``docling``, whose model behaviour changes parse output, and T0.9's
bake-off numbers depend on that output. ``make lock-help`` documents the
workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Fixed namespace for the deterministic serial number. A random UUID would
#: make two runs over an unchanged tree produce different bytes, and an SBOM
#: that changes on every run cannot be diffed between releases — which is the
#: only reason anyone diffs one.
_SERIAL_NAMESPACE = uuid.UUID("6f1d4c1a-9a2e-5b7c-8d3f-0a1b2c3d4e5f")


# --------------------------------------------------------------------------
# SBOM
# --------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """PEP 503 normalisation, which is also what a purl wants.

    Module-level rather than a method because :func:`read_locked_versions` keys
    ``uv.lock``'s names with the same rule; two spellings of "is this the same
    distribution" is how a lock entry ends up looking absent.
    """
    return "".join(
        char if char.isalnum() or char in {".", "-", "_"} else "-" for char in str(name).lower()
    ).replace("_", "-")


@dataclass(frozen=True, slots=True)
class DeclaredDependency:
    """One entry from ``pyproject.toml``'s dependency list.

    ``group`` is ``"runtime"`` for the base dependencies and the extra's own
    name otherwise, which is what lets a consumer answer "is docling in scope
    for this build?" without parsing the requirement strings.
    """

    name: str
    specifier: str
    group: str
    optional: bool = False

    @property
    def normalized_name(self) -> str:
        """PEP 503 normalisation, which is also what a purl wants."""
        return _normalize_name(self.name)

    @property
    def purl(self) -> str:
        # No version qualifier: the specifier is a *range*, and a purl carrying
        # a range in its version field would be read by a scanner as a resolved
        # version. The range is in `properties` where it cannot be mistaken.
        return f"pkg:pypi/{self.normalized_name}"


def _parse_requirement(raw: str) -> str | None:
    """The distribution name from a PEP 508 requirement string.

    Handles the extras-in-brackets form (``uvicorn[standard]>=0.30.0``) by
    taking the part before the bracket, and the environment-marker form by
    taking the part before the semicolon. Returns ``None`` for a VCS or URL
    requirement, which has no distribution name to report.
    """
    text = str(raw or "").strip()
    if not text or text.startswith(("-e", "--", "git+", "http://", "https://", ".")):
        return None
    text = text.split(";", 1)[0].split("[", 1)[0].strip()
    for separator in ("===", "~=", "==", ">=", "<=", "!=", ">", "<"):
        text = text.split(separator, 1)[0].strip()
    return text or None


def read_declared_dependencies(pyproject: Path = PROJECT_ROOT / "pyproject.toml") -> tuple[
    list[DeclaredDependency], dict[str, Any]
]:
    """Every declared dependency, plus the project's own name and version.

    ``tomllib`` rather than a regex over the file. The base list and the extras
    share nothing structurally that a regex could rely on, and ``pyproject.toml``
    is the one file in this repository whose exact syntax matters.
    """
    import tomllib  # noqa: PLC0415 - 3.11+; the project requires >= 3.10 but runs 3.13

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project") or {}
    extras = data.get("project", {}).get("optional-dependencies") or {}

    found: list[DeclaredDependency] = []
    for raw in project.get("dependencies") or ():
        name = _parse_requirement(raw)
        if name:
            found.append(DeclaredDependency(name, str(raw).strip(), "runtime"))
    for group, requirements in sorted(extras.items()):
        for raw in requirements or ():
            name = _parse_requirement(raw)
            if name:
                found.append(
                    DeclaredDependency(name, str(raw).strip(), group, optional=True)
                )

    # A name declared twice (base and extra) is one component with two groups,
    # not two components: CycloneDX identifies a component by bom-ref and two
    # entries with the same purl would be a duplicate the spec forbids.
    merged: dict[str, DeclaredDependency] = {}
    for item in found:
        existing = merged.get(item.normalized_name)
        if existing is None:
            merged[item.normalized_name] = item
            continue
        merged[item.normalized_name] = DeclaredDependency(
            name=existing.name,
            specifier=existing.specifier,
            group=existing.group,
            optional=existing.optional and item.optional,
        )
    return list(merged.values()), {
        "name": str(project.get("name") or "paperwhisperer"),
        "version": str(project.get("version") or "0.0.0"),
        "description": str(project.get("description") or ""),
        "requires_python": str(project.get("requires-python") or ""),
        "extras": sorted(extras),
    }


def _installed_version(name: str) -> str:
    """What is installed *here*, or ``""``.

    Reported as a property and never as ``component.version``, because it
    describes the machine that generated the SBOM rather than the artifact the
    SBOM describes. On a CI runner without the ``parse`` extra it is empty for
    docling, which is exactly the discrepancy a real scanner would need to know
    about and is precisely what a lockfile would have made impossible.
    """
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version(name)
    except Exception:  # noqa: BLE001 - including PackageNotFoundError
        return ""


#: The version lock the project committed to. Read, never written: resolving is
#: ``uv lock``'s job, and running an evaluation should not repin anything.
LOCKFILE = PROJECT_ROOT / "uv.lock"

#: Property name carrying the lock's answer. A constant because the tests and
#: ``sbom_limitations`` both have to name it, and a string spelled twice is a
#: string that eventually disagrees with itself.
LOCKED_VERSION_PROPERTY = "pw:locked_version"


def read_locked_versions(lockfile: Path = LOCKFILE) -> dict[str, tuple[str, ...]]:
    """``{normalized name: (version, ...)}`` from ``uv.lock``; ``{}`` if absent.

    A tuple rather than one string because the lock resolves *per Python
    version*: ``numpy`` appears three times, selected by ``resolution-markers``,
    and collapsing that to a single number would invent a pin the lock does not
    make. Virtually every distribution has exactly one entry.

    A missing or unreadable lock yields an empty map rather than an exception.
    The SBOM has a documented answer for an unlocked tree (``not-locked``), and
    a generator that refuses to start without a lock cannot tell you *that* you
    have none -- which is the one thing it is supposed to be able to say.
    """
    import tomllib  # noqa: PLC0415 - 3.11+; the project requires >= 3.10 but runs 3.13

    try:
        data = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    found: dict[str, set[str]] = {}
    for entry in data.get("package") or ():
        name = entry.get("name")
        version = entry.get("version")
        if name and version:
            found.setdefault(_normalize_name(name), set()).add(str(version))
    return {name: tuple(sorted(versions)) for name, versions in found.items()}


def build_sbom(pyproject: Path = PROJECT_ROOT / "pyproject.toml") -> dict[str, Any]:
    """A CycloneDX 1.5 SBOM over the declared dependencies.

    ``serialNumber`` is derived from the content rather than generated, so two
    runs over an unchanged tree produce identical bytes and a changed tree
    produces a different one. A random serial would make every release look
    like a dependency change.
    """
    dependencies, project = read_declared_dependencies(pyproject)
    locked = read_locked_versions()
    root_ref = f"pkg:pypi/{project['name']}@{project['version']}"

    components: list[dict[str, Any]] = []
    for item in sorted(dependencies, key=lambda entry: entry.normalized_name):
        installed = _installed_version(item.normalized_name)
        components.append({
            "type": "library",
            "bom-ref": item.purl,
            "name": item.normalized_name,
            "version": item.specifier,
            "purl": item.purl,
            "properties": [
                {"name": "pw:declared_specifier", "value": item.specifier},
                {"name": "pw:dependency_group", "value": item.group},
                {"name": "pw:optional", "value": "true" if item.optional else "false"},
                {
                    "name": "pw:environment_resolved_version",
                    "value": installed or "not-installed",
                },
                {
                    # What the lock resolved *for this repository*, which is a
                    # different fact from the floor above it and from what the
                    # generating machine happens to have. Comma-separated when
                    # the lock carries several entries selected by
                    # `resolution-markers`; `not-locked` when there is no lock
                    # or the distribution is not in it.
                    "name": LOCKED_VERSION_PROPERTY,
                    "value": ",".join(locked.get(item.normalized_name, ())) or "not-locked",
                },
            ],
        })

    payload_for_digest = json.dumps(
        [[item.normalized_name, item.specifier, item.group]
         for item in sorted(dependencies, key=lambda entry: entry.normalized_name)],
        sort_keys=True,
    )
    digest = hashlib.sha256(payload_for_digest.encode("utf-8")).hexdigest()
    serial = f"urn:uuid:{uuid.uuid5(_SERIAL_NAMESPACE, digest)}"

    bom: dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": serial,
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": project["name"],
                "version": project["version"],
                "purl": root_ref,
                "description": project["description"],
                "properties": [
                    {"name": "pw:requires_python", "value": project["requires_python"]},
                    {"name": "pw:extras", "value": ",".join(project["extras"])},
                ],
            },
            "properties": [
                {
                    "name": "pw:sbom_scope",
                    "value": (
                        "declared-only: direct dependencies whose component.version is "
                        "the >= specifier, with the uv.lock resolution reported as "
                        "pw:locked_version; no transitive closure, no hashes, no licences"
                    ),
                },
            ],
        },
        "components": components,
        "dependencies": [
            {"ref": root_ref, "dependsOn": [item.purl for item in dependencies]},
            *[
                {"ref": component["bom-ref"], "dependsOn": []}
                for component in components
            ],
        ],
    }
    return bom


def sbom_limitations(bom: dict[str, Any]) -> list[str]:
    """What a real scanner would add that this cannot. Enumerated, not implied."""
    return [
        f"只声明 {len(bom.get('components') or ())} 个直接依赖，"
        "没有传递闭包——一个 Python 环境的实际成分远不止这些；"
        "完整的闭包在 uv.lock 里，本工具只展开 pyproject.toml 声明的那一层。",
        "component.version 是 pyproject.toml 里的 >= 下限，不是某个部署装到的东西。"
        "仓库的 lockfile（uv.lock）解析结果记在 pw:locked_version 属性里，"
        "但那描述的是本仓库的解析环境，不是产物落地的环境；"
        "把两者分开读是对的，混起来做 CVE 匹配仍会得到不精确的结果。",
        "环境实际安装的版本记录在 pw:environment_resolved_version 属性里，"
        "它描述的是生成 SBOM 的这台机器，不是产物要投放的环境；"
        "例如本机没有安装 docling（parse extra），该属性即为 not-installed。",
        "没有许可证信息：pyproject.toml 未逐个声明依赖的许可证，uv.lock 里也没有，"
        "此处不臆造。",
        "没有完整性哈希（sha256 of the wheel/sdist）：uv.lock 里每个包都有，"
        "但那是锁定时刻公开产物的哈希，不是本次部署实际安装的字节；"
        "搬进来会被读成「已校验」，而它没有校验任何东西。",
        "没有漏洞数据：cyclonedx-py / osv-scanner 才做这一步，"
        "而它们匹配的是解析后的版本，本文件的 pw:locked_version 只是它的一个输入。",
        "没有文件级清单（file-level SBOM）：本 SBOM 描述的是产品，不是它的源码树。",
    ]


# --------------------------------------------------------------------------
# Injection runner
# --------------------------------------------------------------------------


def cmd_injection(args: argparse.Namespace) -> int:
    from eval.injection import harness  # noqa: PLC0415 - optional, keeps --help cheap

    if args.compromised:
        # The self-check: run the corpus against a model that *does* comply.
        # A harness whose assertions cannot be shown to fire is a harness that
        # has not been tested, and shipping it green would mean nothing.
        suite = harness.run_offline_suite(responder=harness.compliant_transport)
        print("模式: offline + 已被攻陷的模型（用于证明断言会失败）")
    elif args.live:
        suite = harness.run_live_suite()
    else:
        suite = harness.run_offline_suite()

    print(f"语料版本: {suite.corpus_version}")
    print(f"模式: {suite.mode}")
    if suite.skip_reason:
        print(f"跳过原因: {suite.skip_reason}")
    for report in suite.reports:
        marks = "".join(
            {"pass": ".", "fail": "F", "skip": "s", "error": "E", "manual": "?"}.get(
                item.status, "?"
            )
            for item in report.findings
        )
        flag = "FAIL" if report.failures else "ok  "
        print(f"  [{flag}] {report.sample.id:34} {marks}  ({report.sample.vector})")
        for item in report.failures:
            print(f"         - {item.check}: {item.detail}")
    print()
    print(f"样本 {len(suite.reports)} 份，失败 {len(suite.failures)} 项。")
    print("以下问题本 harness 无法自动判定（见 harness.UNJUDGEABLE）：")
    for item in harness.UNJUDGEABLE:
        print(f"  - {item}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(suite.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nJSON 报告已写入 {args.out}")
    return 1 if suite.failures else 0


def cmd_sbom(args: argparse.Namespace) -> int:
    bom = build_sbom(Path(args.pyproject))
    text = json.dumps(bom, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"SBOM 已写入 {args.out}（{len(bom['components'])} 个组件，"
              f"serialNumber={bom['serialNumber']}）", file=sys.stderr)
    else:
        # The document on stdout, the commentary on stderr. A tool whose
        # primary output is piped into a scanner has to keep them separate: a
        # human-readable paragraph after the closing brace makes the stream
        # unparseable JSON, and `sbom > bom.json` would silently produce a file
        # nothing can read.
        sys.stdout.write(text)
    if not args.quiet:
        print(file=sys.stderr)
        print("这份 SBOM 缺什么（cyclonedx-py 能给而它给不了的）：", file=sys.stderr)
        for item in sbom_limitations(bom):
            print(f"  - {item}", file=sys.stderr)
        print(file=sys.stderr)
        print("依赖锁定：版本锁是 uv.lock，本工具只读不写。"
              "刷新它用 `uv lock`，不要在跑评测时顺手重建——那会重新固定 docling，"
              "进而改动解析输出与 T0.9 的 bake-off 数字。见 `make lock-help`。",
              file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_injection_eval",
        description="T4.2 安全评测：文档 prompt injection 检测与 SBOM 生成。",
    )
    sub = parser.add_subparsers(dest="command")

    injection = sub.add_parser("injection", help="运行 prompt injection 语料")
    injection.add_argument(
        "--live", action="store_true",
        help="额外对真实 LLM 端点发起调用（未配置端点时干净跳过）",
    )
    injection.add_argument(
        "--compromised", action="store_true",
        help="用已被攻陷的模型跑一遍，用来证明断言会失败（自检）",
    )
    injection.add_argument("--out", default="", help="把 JSON 报告写到该路径")
    injection.set_defaults(func=cmd_injection)

    sbom = sub.add_parser("sbom", help="从 pyproject.toml 生成 CycloneDX 1.5 SBOM")
    sbom.add_argument(
        "--pyproject", default=str(PROJECT_ROOT / "pyproject.toml"), help="pyproject.toml 路径"
    )
    sbom.add_argument("--out", default="", help="写到该路径而不是标准输出")
    sbom.add_argument("--quiet", action="store_true", help="不打印 lockfile 说明")
    sbom.set_defaults(func=cmd_sbom)

    args = parser.parse_args(argv)
    if not getattr(args, "command", ""):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
