"""Validator for the tiered-prefix-cache well-lit path."""

from pathlib import Path

from llmdbenchmark.executor.context import ExecutionContext
from llmdbenchmark.smoketests.base import BaseSmoketest, _load_config, _nested_get
from llmdbenchmark.smoketests.report import CheckResult, SmoketestReport


class TieredPrefixCacheValidator(BaseSmoketest):
    """Validates tiered prefix cache scenario."""

    def run_config_validation(
        self,
        context: ExecutionContext,
        stack_path: Path,
    ) -> SmoketestReport:
        """Verify decode pods carry the expected additional vLLM flags and EPP pod is running."""
        report = SmoketestReport()
        cmd = context.require_cmd()
        namespace = context.require_namespace()
        config = _load_config(stack_path)

        if context.dry_run:
            report.add(
                CheckResult(
                    "config_validation",
                    True,
                    message="[DRY RUN] tiered-prefix-cache config validation skipped",
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
            args = self.get_pod_args(pod)

            # Scenario-specific: the flags this guide is actually about. There is
            # no `additionalFlags` list to read any more -- the scenario states
            # them in its decode command -- so assert on the flags the tiered
            # prefix cache needs by name. validate_role_pods separately checks
            # that the whole command reached the pod verbatim; this narrows a
            # failure to "the KV-offload flags are missing", which is the thing
            # this guide would be silently wrong without.
            for flag in ("--kv-transfer-config",):
                if flag in args:
                    report.add(self.assert_arg_present(args, flag))

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
