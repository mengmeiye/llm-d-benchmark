"""``fma.launcher.tensorParallelSize`` is a pod-spec fact, not an engine flag.

On the Fast Model Actuation path the launcher process *is* the vLLM entrypoint,
so its flags are stated once, verbatim, in ``fma.launcher.options`` -- the FMA
counterpart of a role's ``engine.command``. Nothing assembles them.

``tensorParallelSize`` survives alongside that string for one reason the string
cannot cover: TP workers exchange over a shared-memory message queue, so a pod
running TP>1 needs a 16Gi in-memory ``/dev/shm`` emptyDir (the pod default of
64Mi crashes NCCL at init). That is a volume, decided before the process exists,
which is exactly the kind of fact Kubernetes has to be told.

These tests pin both halves: the options string reaches the CRD untouched, and
the volume appears only when the width calls for it.
"""

from __future__ import annotations

import yaml
from jinja2 import Environment

from llmdbenchmark.parser.render_plans import RenderPlans

_TEMPLATE_PATH = "config/templates/jinja/24_fma-deployment.yaml.j2"


def _render(values: dict) -> list[dict]:
    env = Environment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )
    env.filters["toyaml"] = RenderPlans._toyaml_filter
    with open(_TEMPLATE_PATH, encoding="utf-8") as fh:
        template = env.from_string(fh.read())
    out = template.render(**values)
    return [yaml.safe_load(doc) for doc in out.split("\n---\n") if doc.strip()]


def _base_values(tensor_parallel_size: int = 1, options: str | None = None) -> dict:
    return {
        "model_id_label": "qwen3-32b-abc123",
        "model": {
            "name": "Qwen/Qwen3-32B",
            "path": "models/Qwen/Qwen3-32B",
            "maxModelLen": 32768,
            "gpuMemoryUtilization": 0.95,
        },
        "namespace": {"name": "bench"},
        "labels": {"inferenceServing": "true"},
        "huggingface": {"enabled": False},
        "scenarioName": "test-scenario",
        "fma": {
            "enabled": True,
            "modelMountPath": "/model-cache",
            "modelPvcName": "model-pvc",
            "mountModelVolume": True,
            "launcher": {
                "options": (
                    "--model Qwen/Qwen3-32B --enable-sleep-mode"
                    if options is None
                    else options
                ),
                "tensorParallelSize": tensor_parallel_size,
                "maxInstances": 4,
                "image": {
                    "repository": "example.com/launcher",
                    "tag": "v0.6.5",
                    "pullPolicy": "IfNotPresent",
                },
                "podTemplate": {"metadata": {}},
                "customPreprocessCommands": [],
            },
            "launcherConfigurator": {"port": 8001},
            "requester": {
                "image": {"repository": "example.com/requester", "tag": "v0.6.5"},
                "probePort": 8080,
                "spiPort": 8081,
                "limitsGPU": 2,
                "limitsCPU": "1",
                "limitsMemory": "250Mi",
                "replicas": 0,
            },
        },
    }


def _options(docs: list[dict]) -> str:
    isc = next(d for d in docs if d and d.get("kind") == "InferenceServerConfig")
    return isc["spec"]["modelServerConfig"]["options"]


def _launcher_pod_spec(docs: list[dict]) -> dict:
    lc = next(d for d in docs if d and d.get("kind") == "LauncherConfig")
    return lc["spec"]["podTemplate"]["spec"]


class TestFmaLauncherOptions:
    def test_options_reach_the_crd_verbatim(self):
        """Whatever the user wrote is what vLLM gets -- flag for flag."""
        written = (
            "--model /model-cache/models/Qwen/Qwen3-32B "
            "--served-model-name Qwen/Qwen3-32B "
            "--enable-sleep-mode --tensor-parallel-size 2 "
            "--max-model-len 32768 --block-size 64"
        )
        assert _options(_render(_base_values(options=written))) == written

    def test_nothing_is_appended_to_the_options(self):
        """No flag is added on the user's behalf, not even the model."""
        assert _options(_render(_base_values(options="--enable-sleep-mode"))) == (
            "--enable-sleep-mode"
        )

    def test_tensor_parallel_size_adds_no_flag(self):
        """The width provisions a volume; the flag is the user's to write."""
        options = _options(_render(_base_values(tensor_parallel_size=2)))

        assert "--tensor-parallel-size" not in options


class TestFmaLauncherSharedMemory:
    def test_default_tp_renders_no_dshm(self):
        spec = _launcher_pod_spec(_render(_base_values()))
        volume_names = {v["name"] for v in spec.get("volumes", [])}
        mount_names = {
            m["name"]
            for c in spec.get("containers", [])
            for m in c.get("volumeMounts", [])
        }
        assert "dshm" not in volume_names
        assert "dshm" not in mount_names

    def test_tp_greater_than_one_provisions_dshm(self):
        spec = _launcher_pod_spec(_render(_base_values(tensor_parallel_size=2)))
        dshm_vol = next(v for v in spec["volumes"] if v["name"] == "dshm")
        assert dshm_vol["emptyDir"]["medium"] == "Memory"
        assert dshm_vol["emptyDir"]["sizeLimit"] == "16Gi"
        dshm_mount = next(
            m
            for c in spec["containers"]
            for m in c.get("volumeMounts", [])
            if m["name"] == "dshm"
        )
        assert dshm_mount["mountPath"] == "/dev/shm"
