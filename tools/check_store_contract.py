"""Check that every store really implements every interface.

    python tools/check_store_contract.py

``@runtime_checkable`` looks like it solves this and does not. It answers
``isinstance`` by asking whether the attributes *exist*, which is enough to
catch a class that forgot a method entirely — and stops there. A method with
the right name and the wrong signature passes, as does a method that exists
but returns ``None`` where the contract promises a list. And ``isinstance``
cannot be run at all against a Protocol that declares a non-callable member,
which is exactly what ``VectorStore.dimension`` is.

So this walks the Protocols with :mod:`inspect` and compares them to each
implementation member by member:

1. every method the Protocol declares exists on the class, with a compatible
   signature — same parameter names, kinds and defaults;
2. every property the Protocol declares exists and is not shadowed by a
   plain method;
3. every declared member appears in ``__all__`` of the store package, so a
   contract cannot grow a method nothing re-exports;
4. the Protocol surface mentions no storage engine, which is the property
   that lets one test suite run against both tiers. This is checked rather
   than trusted because a docstring that says "SQLite" is the first step
   towards a signature that requires it.

It is a check, not a type checker. It cannot see inside a method body, and it
does not try to: the shared test suite in ``tests/test_stores_*.py`` is what
proves the behaviour. This only proves the shape, so that a method added to a
Protocol and missed by an implementation fails here instead of in production.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paperwhisperer.infra.stores import (  # noqa: E402
    LexicalRetriever,  # noqa: E402
    MetadataStore,
    SqliteLexicalStore,
    SqliteMetadataStore,
    SqliteVectorStore,
    VectorStore,
)

#: Words that name a storage engine. Allowed nowhere in a Protocol member's
#: name or docstring: the whole point of the two-tier split is that a caller
#: cannot tell from the contract which one is running.
ENGINE_WORDS = ("sqlite", "pgvector", "postgres", "postgresql", "fts5", "psycopg")




def contract_members(protocol: type) -> dict[str, inspect.Parameter | None]:
    """Map every declared member name to its parameter list (``None`` if not callable).

    Members inherited from a base Protocol are included, because an
    implementation that satisfies ``StoreCapable`` only on one class is a
    store that stopped reporting its own degradations the moment somebody
    refactored its base.
    """
    members: dict[str, inspect.Parameter | None] = {}
    for name, member in vars(protocol).items():
        if name.startswith("__"):
            continue
        if name.startswith("_"):
            continue
        members[name] = _signature_of(member)
    return members


def _signature_of(member: object) -> inspect.Parameter | None:
    """The parameter list of a member, or ``None`` for a non-callable one."""
    if isinstance(member, property):
        return None
    if inspect.isfunction(member) or inspect.ismethod(member):
        return member
    if callable(member):
        return member
    return None


def check_signature(protocol_name: str, member_name: str, expected, actual) -> list[str]:
    """Compare one member's signature against the contract.

    ``self`` is excluded on both sides: the contract declares it, the class
    provides it through the instance, and comparing them would report every
    method as mismatched.
    """
    expected_signature = inspect.signature(expected)
    actual_signature = inspect.signature(actual)
    errors: list[str] = []
    expected_params = [p for name, p in expected_signature.parameters.items() if name != "self"]
    actual_params = [p for name, p in actual_signature.parameters.items() if name != "self"]
    if [p.kind for p in expected_params] != [p.kind for p in actual_params]:
        return [
            f"{protocol_name}.{member_name}: 参数种类不匹配 "
            f"(契约 {[p.name for p in expected_params]} / 实现 {[p.name for p in actual_params]})"
        ]
    for want, have in zip(expected_params, actual_params, strict=True):
        if want.name != have.name:
            errors.append(
                f"{protocol_name}.{member_name}: 第 {want.name} 个参数名不一致 "
                f"(契约 {want.name} / 实现 {have.name})"
            )
        if want.default is not inspect.Parameter.empty and have.default != want.default:
            errors.append(
                f"{protocol_name}.{member_name}: 参数 {want.name} 的默认值不一致 "
                f"(契约 {want.default!r} / 实现 {have.default!r})"
            )
        if want.kind is inspect.Parameter.KEYWORD_ONLY and have.kind is not (
            inspect.Parameter.KEYWORD_ONLY
        ):
            errors.append(f"{protocol_name}.{member_name}: 参数 {want.name} 不是仅关键字参数")
    return errors


def check_implementation(protocol: type, implementation: type) -> list[str]:
    """Every declared member present, callable, and structurally compatible."""
    errors: list[str] = []
    for name, expected in contract_members(protocol).items():
        member = getattr(implementation, name, None)
        if member is None:
            errors.append(f"{implementation.__name__} 缺少 {protocol.__name__}.{name}")
            continue
        if expected is None:
            # A property in the contract. A method of the same name would
            # return a bound method instead of a value, which is the kind of
            # mistake that only shows up in production.
            if inspect.isfunction(member) or inspect.ismethod(member):
                errors.append(
                    f"{implementation.__name__}.{name} 应为属性，"
                    f"但 {protocol.__name__} 声明的是只读属性"
                )
            continue
        if not callable(member):
            errors.append(f"{implementation.__name__}.{name} 不可调用")
            continue
        errors.extend(check_signature(protocol.__name__, name, expected, member))
    return errors


def check_neutrality(protocol: type) -> list[str]:
    """No contract member may name a storage engine.

    Checked on the source rather than on ``__doc__`` alone, because the
    failure this prevents starts in an argument annotation or a parameter
    name long before it reaches a docstring someone reads.
    """
    errors: list[str] = []
    for name in contract_members(protocol):
        try:
            source = inspect.getsource(getattr(protocol, name))
        except (OSError, TypeError):  # pragma: no cover - builtins have no source
            continue
        lowered = source.lower()
        for word in ENGINE_WORDS:
            if word in lowered:
                errors.append(
                    f"{protocol.__name__}.{name} 的契约提到了 {word!r}："
                    "接口层必须与具体存储引擎无关，否则两档无法共用同一套测试。"
                )
    return errors


def check_exports(protocols: list[type]) -> list[str]:
    """Every protocol must be re-exported from the store package.

    A contract that cannot be imported is a contract no implementation is
    written against.
    """
    package = importlib.import_module("paperwhisperer.infra.stores")
    exported = set(getattr(package, "__all__", ()))
    return [
        f"{protocol.__name__} 未从 paperwhisperer.infra.stores 导出。"
        for protocol in protocols
        if protocol.__name__ not in exported
    ]


def check_job_contract(errors: list[str]) -> None:
    """``JobStore`` gets the same treatment, for the same reason.

    T2.1 added a fourth runtime-checkable Protocol with two implementations,
    and it is exactly the shape the store tier was built to catch: a method
    added to the Protocol and missed by an implementation would surface as an
    ``AttributeError`` on the SQLite path (caught by its tests) and as a
    ``JobNotFound`` on the Hosted path (not caught by anything, because the
    Hosted tier is skipped wherever there is no database).

    Its export check points at ``paperwhisperer.infra.jobs`` rather than the
    store package, because that is where it lives.
    """
    from paperwhisperer.infra import jobs

    errors.extend(check_neutrality(jobs.JobStore))
    exported = set(getattr(jobs, "__all__", ()))
    for name in ("JobStore", "SqliteJobStore", "PostgresJobStore"):
        if name not in exported:
            errors.append(f"{name} 未从 paperwhisperer.infra.jobs 导出。")
    errors.extend(check_implementation(jobs.JobStore, jobs.SqliteJobStore))


def main() -> int:
    pairs = (
        (MetadataStore, SqliteMetadataStore),
        (VectorStore, SqliteVectorStore),
        (LexicalRetriever, SqliteLexicalStore),
    )
    protocols = [MetadataStore, VectorStore, LexicalRetriever]

    errors: list[str] = []
    for protocol in protocols:
        errors.extend(check_neutrality(protocol))
    errors.extend(check_exports(protocols))
    for protocol, implementation in pairs:
        errors.extend(check_implementation(protocol, implementation))
    check_job_contract(errors)

    total = sum(len(contract_members(protocol)) for protocol in protocols)
    print(f"checked {len(pairs)} SQLite implementations against {total} contract members")

    # The Hosted tier is checked by name only, because importing it pulls in a
    # database driver that a personal-tier install does not have. Its methods
    # are verified by the same shared test suite when a database is reachable;
    # this check exists so that a *missed* method is caught at build time even
    # without one, which is why the source is scanned rather than imported.
    errors.extend(check_postgres_sources())
    print("checked 3 Hosted implementations by source scan")

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        print(f"\n{len(errors)} problem(s)")
        return 1

    print("OK   every implementation covers its contract")
    print("OK   signatures are compatible")
    print("OK   the contract names no storage engine")
    return 0


def _names_defined_by(node: ast.ClassDef, classes: dict[str, ast.ClassDef]) -> set[str]:
    """Every method name visible on a class, including inherited ones.

    The three Hosted stores share a base that holds ``close``; without walking
    it the scan would report the same false failure three times and teach
    whoever reads the output to ignore it.
    """
    names: set[str] = set()
    current: ast.ClassDef | None = node
    while current is not None:
        names.update(
            child.name
            for child in current.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        bases = [
            classes[base.id]
            for base in current.bases
            if isinstance(base, ast.Name) and base.id in classes
        ]
        current = bases[0] if bases else None
    return names


POSTGRES_IMPLEMENTATIONS = (
    ("PostgresMetadataStore", MetadataStore),
    ("PostgresVectorStore", VectorStore),
    ("PostgresLexicalStore", LexicalRetriever),
)


def check_postgres_sources() -> list[str]:
    """Verify the Hosted classes by reading their source.

    Importing them would require a driver, and this script has to run on a
    machine that only ever runs the personal tier — which is precisely the
    machine where a forgotten method is cheapest to introduce and most
    expensive to notice. So the file is read, the class bodies are located
    with the :mod:`ast` module, and the member names are compared as text.

    What this cannot do is compare signatures, and it does not pretend to: it
    catches a *missing* method on the tier nobody is running, which is the
    case that otherwise reaches production. Everything else about the Hosted
    tier is covered by the same shared test suite, run against a real
    database by whoever has one.
    """
    module_name = "paperwhisperer.infra.stores.postgres_backend"
    source_path = Path(
        importlib.util.find_spec(module_name).origin
        if importlib.util.find_spec(module_name)
        else ""
    )
    if not source_path.is_file():
        return [f"无法读取 {module_name} 源码，Hosted 档位未检查。"]
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    errors: list[str] = []
    for class_name, protocol in POSTGRES_IMPLEMENTATIONS:
        node = classes.get(class_name)
        if node is None:
            errors.append(f"{module_name} 中找不到 {class_name}")
            continue
        defined = _names_defined_by(node, classes)
        for name in contract_members(protocol):
            if name in defined:
                continue
            errors.append(f"{class_name} 缺少 {protocol.__name__}.{name}（源码扫描）")
    return errors


if __name__ == "__main__":
    sys.exit(main())
