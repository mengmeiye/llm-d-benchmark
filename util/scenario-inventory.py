#!/usr/bin/env python3
"""List what each scenario deploys: which engine, which deploy method.

This is a *reading* tool -- it opens the scenario files, never the cluster, and
answers the questions a test run needs before it starts:

  * **which inference engine does each role launch?**  Read out of the role's
    verbatim ``engine.command`` with :func:`llmdbenchmark.engine.parse_command`,
    the same way the renderer reads it. No engine list is maintained here: add
    an ``EngineSpec`` and the scenarios using it are classified by it.
  * **which deploy method is the scenario written for?**  Read off the
    ``enabled:`` flags, with defaults.yaml supplying the ones a scenario omits.
  * **which ``--spec`` name runs it?**  A file under config/specification/ says
    which scenario it reads, so the two spellings can disagree; the name the CLI
    actually accepts is reported (``-`` when no specification reads the file at
    all, which means nothing can run that scenario).
  * **which engines can it be switched to without editing it?**  A scenario may
    carry a commented-out launch command for another engine, tagged
    ``# @engine <name>``. Those tags are reported as the stack's
    *alternatives*, and ``--apply <name>`` performs the switch -- see
    :func:`apply_alternative`.

so a test run can select by engine or by method instead of carrying
hand-maintained scenario lists that go stale as scenarios are added.

Where a role's engine comes from is reported too, because it changes what a
test proves:

  ``vllm``            the scenario states a ``vllm serve`` line; that line runs.
  ``vllm(default)``   the scenario states no command, so the role inherits the
                      minimal launch line from defaults.yaml.
  ``sglang(kust)``    the kustomize method: the upstream llm-d guide supplies
                      the command, and ``kustomize.acceleratorBackend`` picks
                      the engine.
  ``image-default``   the command is explicitly empty -- the image's own
                      entrypoint launches the server (llm-d-inference-sim).

Usage:
  util/scenario-inventory.py                        # every stack, as a table
  util/scenario-inventory.py --engine sglang        # only SGLang stacks
  util/scenario-inventory.py --method standalone --format specs
  util/scenario-inventory.py --alternative trtllm   # switchable to TRT-LLM
  util/scenario-inventory.py --select guides/optimized-baseline --apply sglang
  util/scenario-inventory.py --json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402  (after sys.path, so the repo is importable)

from llmdbenchmark.engine import (  # noqa: E402
    ENGINE_ROLES,
    alternative_engines,
    apply_alternative,
    get_engine_spec,
    is_known_engine,
    known_engines,
    parse_command,
)

SCENARIO_DIR = REPO / "config" / "scenarios"
SPEC_DIR = REPO / "config" / "specification"
DEFAULTS = REPO / "config" / "templates" / "values" / "defaults.yaml"

#: How a specification names the scenario file it drives.
SCENARIO_REF = re.compile(r"config/scenarios/(\S+\.yaml)")

#: Deploy methods, spelled as ``-t`` accepts them.
METHODS = ("modelservice", "standalone", "kustomize", "nok8s", "fma")

#: Methods that deploy each engine role, so a stack's engines reflect what will
#: actually run rather than what a disabled alternative happens to carry.
ROLES_BY_METHOD = {
    "modelservice": ("decode", "prefill"),
    "fma": ("decode", "prefill"),
    "standalone": ("standalone",),
    "nok8s": ("nok8s",),
    "kustomize": (),  # the guide's own manifests; see acceleratorBackend
}

#: An explicitly empty command: the image's entrypoint is the launcher.
IMAGE_DEFAULT = "image-default"


def spec_names() -> dict[str, list[str]]:
    """Map each scenario file to the ``--spec`` names that drive it.

    `--spec` names a file under config/specification/, and that file says which
    scenario it reads. The two names usually match, but they are separate files
    and a rename can leave them disagreeing -- so the name the CLI accepts is
    read here rather than guessed from the scenario's own path.
    """
    found: dict[str, list[str]] = {}
    for path in sorted(SPEC_DIR.rglob("*.yaml.j2")):
        name = path.relative_to(SPEC_DIR).name[: -len(".yaml.j2")]
        name = (path.relative_to(SPEC_DIR).parent / name).as_posix().lstrip("./")
        match = SCENARIO_REF.search(path.read_text())
        if match:
            scenario = match.group(1)[: -len(".yaml")]
            found.setdefault(scenario, []).append(name)
    return found


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def dig(source, path: str):
    """Value at dotted ``path`` in ``source``, or None."""
    node = source
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def lookup(*sources, path: str):
    """First non-None value at dotted ``path`` across ``sources``.

    A scenario may state a key at the stack's top level, under ``common:``
    (which the renderer hoists), nested under ``modelservice:``, or -- in a
    multi-stack file -- once in ``shared:``. Reading each in precedence order
    answers a question about the merge without re-implementing it.
    """
    for source in sources:
        value = dig(source, path) if isinstance(source, dict) else None
        if value is not None:
            return value
    return None


def engine_commands(node, trail=()) -> dict[str, str]:
    """Map role name -> its ``engine.command`` text, for every one in ``node``.

    Walks the tree rather than reading fixed paths: a command sits under
    ``modelservice.decode.engine.command`` in one scenario and
    ``decode.engine.command`` in the next, and only the nearest enclosing role
    name matters. A command outside any role is keyed ``""`` -- the stack-wide
    fallback a role without its own command inherits.
    """
    found: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "engine" and isinstance(value, dict):
                command = value.get("command")
                if isinstance(command, str):
                    role = next((t for t in reversed(trail) if t in ENGINE_ROLES), "")
                    found.setdefault(role, command)
            found.update(engine_commands(value, (*trail, key)))
    elif isinstance(node, list):
        for item in node:
            found.update(engine_commands(item, trail))
    return found


def image_engine(images) -> str | None:
    """Engine named by an image repository, e.g. ``.../llm-d-inference-sim``.

    Read only when a role's command is explicitly empty: the image's entrypoint
    is then the launcher, and its name is the only statement of which engine
    that is. Repository basenames are matched against the registry's own
    aliases, so nothing engine-specific is spelled here.
    """
    if not isinstance(images, dict):
        return None
    for entry in images.values():
        repo = entry.get("repository") if isinstance(entry, dict) else None
        if isinstance(repo, str) and is_known_engine(repo.rsplit("/", 1)[-1]):
            return get_engine_spec(repo.rsplit("/", 1)[-1]).name
    return None


def role_engine(
    role, commands, declared, has_default_command, images=None
) -> tuple[str, str]:
    """``(engine, source)`` for one role, by the resolver's own precedence.

    ``source`` records where the launch line came from, because a role running
    defaults.yaml's minimal line is not testing the scenario's own command.
    """
    name = declared.get(role) or declared.get("")
    command = commands.get(role)
    source = "scenario"
    if command is None:
        command = commands.get("")
    if command is None:
        # defaults.yaml carries a minimal `vllm serve` line per role, so an
        # unstated command means vLLM with that line, not the image entrypoint.
        command = has_default_command.get(role)
        source = "defaults"
    if isinstance(command, str) and not command.strip():
        named = name or image_engine(images)
        return (get_engine_spec(named).name if named else IMAGE_DEFAULT), "image"

    if command is not None:
        # parse_command, not a bare launcher match: it steps over a preamble
        # (`export ...; source ...;`) exactly as the resolver does.
        parsed = parse_command(command, name)
        if parsed.engine:
            return parsed.engine, source
    if name:
        return (
            get_engine_spec(name).name if is_known_engine(name) else str(name)
        ), "declared"
    return get_engine_spec(None).name, source


def stack_facts(spec, stack, shared, defaults, spec_name="", alternatives=()) -> dict:
    """Everything a test run needs to know about one stack, from its text."""
    commands = engine_commands(stack)
    for role, command in engine_commands(shared).items():
        commands.setdefault(role, command)

    nested = (
        stack.get("modelservice") if isinstance(stack.get("modelservice"), dict) else {}
    )
    declared, roles = {}, {}
    for role in ("", *ENGINE_ROLES):
        prefix = f"{role}." if role else ""
        name = lookup(
            stack, stack.get("common"), nested, shared, path=f"{prefix}engine.name"
        )
        if name:
            declared[role] = str(name)

    default_commands = {
        role: dig(defaults, f"{role}.engine.command") for role in ENGINE_ROLES
    }

    methods = []
    for method in METHODS:
        enabled = lookup(stack, shared, path=f"{method}.enabled")
        if enabled is None:
            enabled = dig(defaults, f"{method}.enabled")
        if enabled:
            methods.append(method)

    live = {r for m in methods for r in ROLES_BY_METHOD[m]}

    guide = lookup(stack, shared, path="kustomize.guideName")
    # Methods `-t` can select on this stack even though the scenario does not
    # enable them. modelservice and standalone are always available (a role with
    # no command of its own inherits defaults.yaml's line); the other three need
    # the scenario to carry their section.
    forcible = ["modelservice", "standalone"]
    if guide:
        forcible.append("kustomize")
    if isinstance(lookup(stack, shared, path="nok8s"), dict):
        forcible.append("nok8s")
    if isinstance(lookup(stack, shared, path="fma"), dict):
        forcible.append("fma")
    forcible = [m for m in METHODS if m in forcible and m not in methods]

    # Roles any selectable method would deploy. A role with no section of its
    # own is still reported: `-t standalone` on a scenario that does not enable
    # standalone runs defaults.yaml's launch line, and a test run needs to know
    # which engine that is.
    reachable = live | {r for m in forcible for r in ROLES_BY_METHOD[m]}
    for role in ENGINE_ROLES:
        section = lookup(stack, nested, shared, path=role)
        enabled = lookup(stack, nested, shared, path=f"{role}.enabled")
        if enabled is None:
            enabled = dig(defaults, f"{role}.enabled")
        if not isinstance(section, dict) and role not in reachable:
            continue
        engine, source = role_engine(
            role,
            commands,
            declared,
            default_commands,
            lookup(stack, stack.get("common"), shared, path="images"),
        )
        roles[role] = {"engine": engine, "source": source, "enabled": bool(enabled)}

    if guide:
        # The upstream guide's manifests define the launch; this key picks which
        # engine's overlay is applied (`gpu/vllm`, `gpu/sglang`). Reported
        # whenever a guide is named, because `-t kustomize` can select this path
        # on a scenario that does not enable it by default.
        backend = (
            lookup(stack, shared, path="kustomize.acceleratorBackend") or "gpu/vllm"
        )
        roles["kustomize"] = {
            "engine": str(backend).rsplit("/", 1)[-1],
            "source": "kustomize",
            "enabled": "kustomize" in methods,
        }

    engines = sorted(
        {
            r["engine"]
            for role, r in roles.items()
            if r["enabled"] and (role == "kustomize" or role in live)
        }
    )

    count = lookup(stack, stack.get("common"), shared, path="accelerator.count")
    return {
        "spec": spec,
        "specName": spec_name,
        "stack": stack.get("name") or "",
        "group": spec.split("/")[0],
        "methods": methods,
        "forcible": forcible,
        "engines": engines or sorted({r["engine"] for r in roles.values()}),
        # Engines the file can be switched to without editing it: one commented
        # command per engine, tagged `# @engine <name>`. `--apply` does the
        # switch; until then these do not run, which is exactly why they are
        # reported -- an untested alternative is the one that rots.
        "alternatives": list(alternatives),
        "accel": "cpu" if str(count) == "0" else "accel",
        "roles": roles,
    }


def inventory() -> list[dict]:
    """Facts for every stack in every scenario file, sorted by spec."""
    defaults = load(DEFAULTS)
    by_scenario = spec_names()
    rows = []
    for path in sorted(SCENARIO_DIR.rglob("*.yaml")):
        spec = path.relative_to(SCENARIO_DIR).with_suffix("").as_posix()
        # No specification reads this file, so there is no --spec value for it.
        spec_name = (by_scenario.get(spec) or [""])[0]
        alternatives = alternative_engines(path.read_text())
        try:
            doc = load(path)
        except yaml.YAMLError as exc:
            rows.append(
                {
                    "spec": spec,
                    "specName": spec_name,
                    "stack": "",
                    "group": spec.split("/")[0],
                    "methods": [],
                    "engines": [],
                    "alternatives": alternatives,
                    "accel": "?",
                    "roles": {},
                    "error": str(exc).splitlines()[0],
                }
            )
            continue
        shared = doc.get("shared") or {}
        for stack in doc.get("scenario") or []:
            if isinstance(stack, dict):
                rows.append(
                    stack_facts(spec, stack, shared, defaults, spec_name, alternatives)
                )
    return rows


SOURCE_MARK = {
    "scenario": "",
    "defaults": "(default)",
    "kustomize": "(kust)",
    "image": "",
    "declared": "(declared)",
}


def role_cell(row: dict, show_disabled: bool = True) -> str:
    parts = []
    for role, info in sorted(row["roles"].items()):
        if not info["enabled"] and not show_disabled:
            continue
        mark = SOURCE_MARK.get(info["source"], "")
        off = "" if info["enabled"] else " off"
        parts.append(f"{role}={info['engine']}{mark}{off}")
    return ",".join(parts) or "-"


def matches(row: dict, args) -> bool:
    if args.engine and not ({e.lower() for e in row["engines"]} & set(args.engine)):
        return False
    if args.method and not (
        (set(row["methods"]) | set(row.get("forcible") or [])) & set(args.method)
    ):
        return False
    if args.group and row["group"] not in args.group:
        return False
    if args.alternative and not (
        {a.lower() for a in row.get("alternatives") or ()} & set(args.alternative)
    ):
        return False
    if args.accel and row["accel"] != args.accel:
        return False
    if args.select and not any(fnmatch.fnmatch(row["spec"], p) for p in args.select):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=(
            "Engine labels: vllm = the scenario's own command; vllm(default) = "
            f"defaults.yaml's minimal line; x(kust) = upstream guide manifests; "
            f"{IMAGE_DEFAULT} = the image entrypoint launches the server.\n"
            "METHOD(S): the methods the scenario enables, then in [brackets] the "
            "ones `-t` can force on it.\n"
            "ALT: engines the scenario carries a commented-out command for "
            "(`# @engine NAME`); --apply NAME prints the file with that command "
            "switched in."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--engine",
        action="append",
        metavar="NAME",
        help="keep stacks running this engine (repeatable). Known: "
        f"{', '.join(known_engines())}, {IMAGE_DEFAULT}",
    )
    parser.add_argument(
        "--method",
        action="append",
        metavar="M",
        help=f"keep stacks that enable this method (repeatable): {', '.join(METHODS)}",
    )
    parser.add_argument(
        "--group",
        action="append",
        metavar="DIR",
        help="keep scenarios under this directory (examples, guides, cicd, experimental)",
    )
    parser.add_argument(
        "--alternative",
        action="append",
        metavar="NAME",
        help="keep scenarios carrying a commented-out command for this engine "
        "(repeatable). See --apply.",
    )
    parser.add_argument(
        "--apply",
        metavar="NAME",
        help="print the selected scenario with its `# @engine NAME` groups "
        "uncommented (and the definitions they replace removed), instead of "
        "listing anything. Exactly one scenario must be selected -- narrow with "
        "--select.",
    )
    parser.add_argument(
        "--accel",
        choices=("cpu", "accel"),
        help="cpu: accelerator.count is 0, so the pods request no device",
    )
    parser.add_argument(
        "--select",
        action="append",
        metavar="GLOB",
        help="keep scenarios whose spec matches this glob (repeatable)",
    )
    parser.add_argument(
        "--format",
        choices=("table", "tsv", "specs", "json"),
        default="table",
        help="table (default), tsv, specs (bare --spec values), json",
    )
    parser.add_argument(
        "--show-disabled",
        action="store_true",
        help="in the table, also show roles the scenario disables",
    )
    parser.add_argument(
        "--json",
        action="store_const",
        const="json",
        dest="format",
        help="shorthand for --format json",
    )
    args = parser.parse_args()

    args.engine = [e.lower() for e in (args.engine or [])]
    args.method = [m.lower() for m in (args.method or [])]
    args.alternative = [a.lower() for a in (args.alternative or [])]
    if args.apply and args.apply.lower() not in args.alternative:
        # `--apply X` means "the scenario that offers X", so it selects too.
        args.alternative.append(args.apply.lower())
    rows = [r for r in inventory() if matches(r, args)]

    if args.apply:
        chosen = list(dict.fromkeys(r["spec"] for r in rows))
        if len(chosen) != 1:
            print(
                f"--apply {args.apply} needs exactly one scenario, "
                f"got {len(chosen)}: {', '.join(chosen) or 'none'}",
                file=sys.stderr,
            )
            return 1
        path = SCENARIO_DIR / f"{chosen[0]}.yaml"
        try:
            sys.stdout.write(apply_alternative(path.read_text(), args.apply))
        except ValueError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    elif args.format == "specs":
        for spec in dict.fromkeys(r["specName"] for r in rows):
            if spec:
                print(spec)
    elif args.format == "tsv":
        for r in rows:
            print(
                "\t".join(
                    [
                        r["spec"],
                        r["stack"],
                        ",".join(r["methods"]) or "-",
                        ",".join(r["forcible"]) or "-",
                        ",".join(r["engines"]) or "-",
                        r["accel"],
                        role_cell(r),
                        r["specName"] or "-",
                        # Appended last on purpose: util/test-scenarios.sh cuts
                        # fields by number, so a new column must not move one.
                        ",".join(r.get("alternatives") or ()) or "-",
                    ]
                )
            )
    else:
        head = ("SCENARIO", "ENGINE(S)", "ALT", "METHOD(S)", "ACCEL", "ROLES")
        cells = [
            (
                r["spec"],
                ",".join(r["engines"]) or "-",
                ",".join(r.get("alternatives") or ()) or "-",
                ",".join(r["methods"])
                + (f" [{','.join(r['forcible'])}]" if r["forcible"] else ""),
                r["accel"],
                role_cell(r, args.show_disabled),
            )
            for r in rows
        ]
        widths = (
            [max(len(c[i]) for c in (head, *cells)) for i in range(len(head))]
            if cells
            else [len(h) for h in head]
        )
        for row in (head, *cells):
            print("  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip())
        print(f"\n{len(cells)} stack(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
