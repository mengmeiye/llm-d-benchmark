"""A model PVC that is too small stops the step before the weights download.

A model PVC outlives teardown on purpose: the next standup finds the weights
already staged instead of pulling them again. It only goes wrong when the next
scenario wants a *bigger* one than the last left behind -- `examples/engines`
takes the 300Gi default and `guides/optimized-baseline` asks for 1Ti -- and
growing it in place is not on offer: a StorageClass need not support expansion,
and another model's weights are sitting in it.

The check for that ran in the right place and reported in the wrong one. It
appended to `errors`, the step carried on, pulled 61 GB of Qwen3-32B into the
undersized volume over 7m41s, and only then failed with a size it had known
since the first second:

    18:27:55  ✅ model PVC "model-pvc": Bound
    18:35:40  ✅ Model download download-model completed (attempt 1)
    18:35:40  ❌ PVC 'model-pvc' exists with size 300Gi but 1Ti is required

Storage this step could not make usable is fatal for everything after it -- the
download Job mounts the volume and so does every engine pod -- so the step now
returns there, and the error says which command clears it.
"""

from __future__ import annotations

import yaml

from llmdbenchmark.executor.command import CommandResult
from llmdbenchmark.executor.context import ExecutionContext
from llmdbenchmark.standup.steps.step_04_model_namespace import ModelNamespaceStep


class _Logger:
    def log_info(self, *a, **k): ...

    def log_warning(self, *a, **k): ...

    def log_error(self, *a, **k): ...


class _Cmd:
    """A cluster that already holds a 300Gi `model-pvc`."""

    def __init__(self, existing: str = "300Gi") -> None:
        self.existing = existing
        self.applied: list[str] = []

    def kube(self, *args, **kwargs) -> CommandResult:
        if args[:3] == ("get", "pvc", "model-pvc"):
            return CommandResult(command="get pvc", exit_code=0, stdout=self.existing)
        if args[0] == "apply":
            self.applied.append(str(args[-1]))
        return CommandResult(command=" ".join(str(a) for a in args), exit_code=0)

    def wait_for_pvc(self, **k) -> CommandResult:
        return CommandResult(command="wait pvc", exit_code=0)

    def wait_for_job(self, **k) -> CommandResult:  # pragma: no cover - must not run
        raise AssertionError("the download Job was launched anyway")


def _context(tmp_path):
    """One `pvc+hf` stack asking for a 1Ti model PVC, with its manifest present."""
    stack = tmp_path / "plan" / "stack01"
    stack.mkdir(parents=True)
    (stack / "config.yaml").write_text(
        yaml.dump(
            {
                "modelservice": {"uriProtocol": "pvc+hf"},
                # Only reached once the storage is accepted, which is how the
                # last test below can tell that it was.
                "control": {"contextSecretName": "llm-d-benchmark-context"},
                "storage": {"modelPvc": {"name": "model-pvc", "size": "1Ti"}},
            }
        ),
        encoding="utf-8",
    )
    (stack / "02_pvc_model-pvc.yaml").write_text("kind: PersistentVolumeClaim\n")
    return ExecutionContext(
        plan_dir=tmp_path / "plan",
        workspace=tmp_path,
        logger=_Logger(),
        namespace="mye",
        rendered_stacks=[stack],
    )


def test_the_step_fails_at_the_storage_not_after_the_download(tmp_path):
    cmd = _Cmd()
    context = _context(tmp_path)
    context.cmd = cmd

    result = ModelNamespaceStep().execute(context)

    assert result.success is False
    assert "skipped the model download" in result.message
    assert any("existing PVC is too small" in e for e in result.errors)
    # The PVC exists, so nothing should have been applied on the way out --
    # least of all a download Job. `_Cmd.wait_for_job` raises if one is.
    assert cmd.applied == [], cmd.applied


def test_the_error_names_the_command_that_clears_it(tmp_path):
    """ "Too small" without a remedy sends the reader looking for a resize."""
    context = _context(tmp_path)
    context.cmd = _Cmd()

    errors = ModelNamespaceStep().execute(context).errors
    remedy = [e for e in errors if "kubectl delete pvc" in e]
    assert remedy, errors
    assert "model-pvc" in remedy[0] and "-n mye" in remedy[0]
    assert "re-downloaded" in remedy[0]


def test_a_big_enough_pvc_is_reused_and_the_step_goes_on(tmp_path):
    """The reason the PVC is left behind at all: skipping the download."""
    cmd = _Cmd(existing="2Ti")
    context = _context(tmp_path)
    context.cmd = cmd

    result = ModelNamespaceStep().execute(context)
    assert not [e for e in result.errors if "too small" in e], result.errors
    # Past the gate, and the PVC was reused rather than re-applied: an existing
    # volume large enough is exactly the case this leaves alone.
    assert "02_pvc_model-pvc.yaml" not in " ".join(cmd.applied)
