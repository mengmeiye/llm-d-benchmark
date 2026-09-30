"""Validator for the precise-prefix-cache-routing well-lit path."""

from pathlib import Path

from llmdbenchmark.executor.context import ExecutionContext
from llmdbenchmark.smoketests.base import BaseSmoketest, _load_config, _nested_get
from llmdbenchmark.smoketests.report import CheckResult, SmoketestReport


class PrecisePrefixCacheAwareValidator(BaseSmoketest):
    """Validates precise prefix cache aware routing scenario."""

    def run_config_validation(
        self,
        context: ExecutionContext,
        stack_path: Path,
    ) -> SmoketestReport:
        """Verify sha256_cbor hash algo, direct vLLM port (no proxy), and EPP pod presence."""
        report = SmoketestReport()
        cmd = context.require_cmd()
        namespace = context.require_namespace()
        config = _load_config(stack_path)

        if context.dry_run:
            report.add(
                CheckResult(
                    "config_validation",
                    True,
                    message="[DRY RUN] precise-prefix-cache-routing config validation skipped",
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
                expected="0",
                actual=str(len(prefill_pods)),
                message=f"{'No' if not prefill_pods else len(prefill_pods)} prefill pod(s) -- decode-only scenario",
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
            args = self.get_pod_args(pod)

            # Scenario-specific: prefix caching hash algo
            report.add(
                self.assert_arg_contains(
                    args, "--prefix-caching-hash-algo", "sha256_cbor"
                )
            )

            # Scenario-specific: this guide runs without the routing sidecar, so
            # the engine must bind the Service port itself -- nothing bridges
            # 8000 to 8200 here. The port in the args is the command's own
            # --port (llm-d-benchmark does not rewrite it); compare it with what
            # the Service exposes.
            engine_port_in_args = None
            if "--port" in args:
                parts = args.split()
                for i, token in enumerate(parts):
                    if token == "--port" and i + 1 < len(parts):
                        engine_port_in_args = parts[i + 1].strip("\\").strip()
                        break

            if engine_port_in_args:
                expected_port = _nested_get(config, "engine", "servicePort")
                if expected_port is not None:
                    report.add(
                        CheckResult(
                            "engine_port",
                            engine_port_in_args
                            in (str(expected_port), "$SERVICE_PORT", "$ENGINE_PORT"),
                            expected=str(expected_port),
                            actual=engine_port_in_args,
                            message=(
                                f"engine binds port {engine_port_in_args} "
                                f"(expected {expected_port} -- no proxy)"
                            ),
                        )
                    )

        # The llm-d-router-{standalone,gateway}-dev charts label the EPP
        # Pod with the mode-specific selector
        # (`llm-d-router-standalone=<release>-epp` or
        # `llm-d-router-gateway=<release>-epp`); they do not apply an
        # `inferencepool=<release>-epp` label. The common
        # `app.kubernetes.io/*` labels only land on the Deployment, not
        # the Pod. We don't know the gateway mode here, so try both --
        # exactly one will match.
        epp_pods: list = []
        for _mode in ("llm-d-router-gateway", "llm-d-router-standalone"):
            epp_pods = self.get_pod_specs(
                cmd,
                namespace,
                f"{_mode}={model_short}-router-epp",
            )
            if epp_pods:
                break
        report.add(
            CheckResult(
                "epp_pod_running",
                len(epp_pods) > 0,
                message=f"EPP pod {'running' if epp_pods else 'not found'}",
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

        return report
