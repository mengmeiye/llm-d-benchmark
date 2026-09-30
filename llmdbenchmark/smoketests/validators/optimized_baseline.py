"""Validator for the optimized-baseline well-lit path."""

from pathlib import Path

from llmdbenchmark.executor.context import ExecutionContext
from llmdbenchmark.smoketests.base import BaseSmoketest, _load_config, _nested_get
from llmdbenchmark.smoketests.report import CheckResult, SmoketestReport
from llmdbenchmark.smoketests.validators.wva import WvaSmoketestMixin


class OptimizedBaselineValidator(WvaSmoketestMixin, BaseSmoketest):
    """Validates optimized baseline scenario.

    Also runs WVA resource checks (controller, KEDA, ScaledObject)
    when the rendered stack has ``wva.enabled: true``; the mixin is a no-op
    for non-WVA stacks.
    """

    def run_config_validation(
        self,
        context: ExecutionContext,
        stack_path: Path,
    ) -> SmoketestReport:
        """Verify decode-only deployment with expected metrics port and shared memory."""
        report = SmoketestReport()
        cmd = context.require_cmd()
        namespace = context.require_namespace()
        config = _load_config(stack_path)

        if context.dry_run:
            report.add(
                CheckResult(
                    "config_validation",
                    True,
                    message="[DRY RUN] optimized-baseline config validation skipped",
                )
            )
            return report

        model_short = (
            config.get("model_id_label", "")
            or _nested_get(config, "model", "shortName")
            or ""
        )

        prefill_pods = self.get_pod_specs(
            cmd,
            namespace,
            f"llm-d.ai/model={model_short},llm-d.ai/role=prefill",
        )
        report.add(
            CheckResult(
                "no_prefill_pods",
                len(prefill_pods) == 0,
                message=f"{'No' if not prefill_pods else len(prefill_pods)} prefill pod(s) -- decode-only",
            )
        )

        decode_pods = self.validate_role_pods(
            cmd,
            namespace,
            config,
            "decode",
            model_short,
            report,
            logger=context.logger,
        )

        if decode_pods:
            pod = decode_pods[0]
            ports = self.get_container_ports(pod)

            # Scenario-specific: the engine also serves /metrics on the port it
            # binds, which is whatever its command's --port said (recorded by
            # resolve_engines as decode.engine.port).
            expected_port = _nested_get(config, "decode", "engine", "port")
            if expected_port is not None:
                expected_port = int(expected_port)
                has_metrics = any(
                    p.get("name") == "metrics"
                    or p.get("containerPort") == expected_port
                    for p in ports
                )
                report.add(
                    CheckResult(
                        "metrics_port",
                        has_metrics,
                        expected=str(expected_port),
                        message=f"Metrics port {expected_port} {'present' if has_metrics else 'not found'}",
                    )
                )

        if decode_pods:
            # Shared memory volume -- only check if scenario defines it
            configured_volumes = _nested_get(config, "engine", "volumes") or []
            configured_vol_names = [
                v.get("name", "") for v in configured_volumes if isinstance(v, dict)
            ]
            if "dshm" in configured_vol_names:
                volumes = self.get_pod_volumes(decode_pods[0])
                report.add(
                    CheckResult(
                        "dshm_volume",
                        "dshm" in volumes,
                        message=f"Shared memory volume 'dshm' {'present' if 'dshm' in volumes else 'not found'}",
                    )
                )

        # WVA checks are a no-op unless the stack has wva.enabled: true
        self.run_wva_checks(context, stack_path, report)

        return report
