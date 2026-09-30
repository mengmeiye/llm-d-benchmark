"""Base smoketest with health checks, inference tests, and pod inspection."""

import base64
import json
import time
from pathlib import Path

from llmdbenchmark.engine import (
    detect_engine,
    get_engine_spec,
    is_known_engine,
    serving_engine,
    serving_port,
    tokenize,
)
from llmdbenchmark.executor.command import CommandExecutor
from llmdbenchmark.executor.context import ExecutionContext
from llmdbenchmark.parser.cluster_resource_resolver import effective_accelerator_count
from llmdbenchmark.smoketests.nok8s import (
    health_check as nok8s_health_check,
    inference_test as nok8s_inference_test,
)
from llmdbenchmark.smoketests.report import CheckResult, SmoketestReport
from llmdbenchmark.utilities.endpoint import (
    _build_overrides,
    _ephemeral_label_args,
    _normalize_url_prefix,
    _rand_suffix,
    compute_gateway_path_prefix,
    find_custom_endpoint,
    find_direct_modelservice_endpoint,
    find_epponly_endpoint,
    find_gateway_endpoint,
    find_kustomize_endpoint,
    find_standalone_endpoint,
    resolve_direct_service_namespace,
    test_model_serving,
)

_RETRYABLE_INDICATORS = ("502", "503", "504", "ServiceUnavailable", "not ready")

# Roles whose pod count is HPA-managed when WVA is enabled. The replica
# count check relaxes from strict equality (== <role>.replicas) to
# range-membership (within wva.hpa.[min,max]Replicas) for these roles only.
#
# Currently only `decode` because the per-stack HPA template
# (28_wva-hpa.yaml.j2) only targets the decode Deployment. If the WVA
# chart grows native prefill autoscaling and we render a second HPA,
# add "prefill" here and the relaxation kicks in automatically.
_WVA_HPA_MANAGED_ROLES: frozenset[str] = frozenset({"decode"})

#: llm-d's engine-neutral name for the serving container (``engine.containerName``
#: in defaults.yaml). A pod running SGLang should not be inspected under the name
#: "vllm", and ``kubectl logs -c modelserver`` reads the same whichever engine the
#: role's command launches.
ENGINE_CONTAINER = "modelserver"

#: Containers that share an engine pod without being the engine. Used only to
#: pick the serving container out of a pod whose engine container carries some
#: other name (a scenario that set ``engine.containerName``, or a manifest from
#: before the rename).
_SIDECAR_CONTAINERS: frozenset[str] = frozenset(
    {"routing-proxy", "istio-proxy", "uds-tokenizer"}
)


def _engine_container(pod_spec: dict) -> str:
    """Name of the serving container in *pod_spec*.

    ``modelserver`` when the pod has it, which is what llm-d-benchmark renders.
    Otherwise the first container that is not a known sidecar, so a pod whose
    engine container was renamed is still inspected rather than checked against
    an empty dict -- a check that reads "flag not found" when the truth is
    "container not found" is worse than no check.
    """
    names = [c.get("name", "") for c in pod_spec.get("spec", {}).get("containers", [])]
    if ENGINE_CONTAINER in names:
        return ENGINE_CONTAINER
    for name in names:
        if name and name not in _SIDECAR_CONTAINERS:
            return name
    return names[0] if names else ENGINE_CONTAINER


def _is_retryable(text: str) -> bool:
    return any(ind in text for ind in _RETRYABLE_INDICATORS) if text else False


def _is_non_transient_error(resp: dict) -> bool:
    if "error" not in resp:
        return False
    error = resp["error"]
    msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
    return not _is_retryable(msg)


class BaseSmoketest:
    """Common validation logic shared by all scenarios.

    Provides health checks, inference testing, pod inspection, and a
    library of assertion helpers that per-scenario validators build on.
    """

    @staticmethod
    def _gateway_path_prefix_for_stack(
        plan_config: dict,
        is_standalone: bool,
        stack_name: str = "",
    ) -> str:
        """Thin wrapper over ``utilities.endpoint.compute_gateway_path_prefix``.

        Kept as a method so subclasses can still override if some future
        validator needs scenario-specific routing logic.
        """
        return compute_gateway_path_prefix(
            plan_config,
            stack_name,
            is_standalone=is_standalone,
        )

    @staticmethod
    def _gateway_routes_health(plan_config: dict) -> bool:
        """Return True when the gateway routes /health to upstream pods.

        A shared HTTPRoute with ``rewriteTo: /`` routes everything, so
        `/{stack-prefix}/health` reaches vLLM's /health at the root. But
        ``rewriteTo: /v1`` (or any non-root path) narrows routing to the
        /v1/* namespace - /health isn't under /v1, so the gateway would
        have no rule matching it and a smoketest probe would 404.

        The smoketest uses this to decide whether to skip the /health
        probe gracefully in shared-HTTPRoute scenarios where the operator
        deliberately chose narrower routing.
        """
        http_route = plan_config.get("httpRoute") or {}
        if http_route.get("mode") != "shared":
            return True  # not a shared route - direct /health works
        rewrite_to = (http_route.get("rewriteTo") or "/").strip()
        # "/" or "" means "rewrite to root" -> /health routes cleanly.
        return rewrite_to in ("", "/")

    @staticmethod
    def discover_endpoint(
        cmd: CommandExecutor,
        context: ExecutionContext,
        plan_config: dict,
    ) -> tuple[str | None, str, bool]:
        """Discover the service/gateway endpoint.

        Returns (service_ip, gateway_port, is_standalone).
        """
        is_standalone = "standalone" in context.deployed_methods
        is_kustomize = "kustomize" in context.deployed_methods
        namespace = context.require_namespace()

        inference_port = serving_port(plan_config)
        release = _nested_get(plan_config, "release") or ""
        gateway_class = _nested_get(plan_config, "gateway", "className") or ""
        model_id_label = plan_config.get("model_id_label", "") or ""

        if is_standalone:
            service_ip, _, gateway_port = find_standalone_endpoint(
                cmd, namespace, inference_port
            )
        elif is_kustomize:
            guide_name = _nested_get(plan_config, "kustomize", "guideName") or ""
            if guide_name:
                service_ip, _, gateway_port = find_kustomize_endpoint(
                    cmd,
                    namespace,
                    guide_name,
                )
            else:
                service_ip, _, gateway_port = find_custom_endpoint(
                    cmd,
                    namespace,
                    "epp",
                )
        elif gateway_class == "epponly":
            # No Kubernetes Gateway -- the EPP service is the data plane.
            # Hit `{model_id_label}-router-epp:80` directly (port 80 is the
            # extraServicePort we add for the standalone chart's Envoy
            # sidecar).
            service_ip, _, gateway_port = find_epponly_endpoint(
                cmd,
                namespace,
                model_id_label,
            )
        elif gateway_class == "none":
            direct_service_namespace = resolve_direct_service_namespace(
                plan_config, namespace
            )
            direct_port = str(
                _nested_get(plan_config, "routing", "servicePort") or "8000"
            )
            service_ip, _, gateway_port = find_direct_modelservice_endpoint(
                cmd,
                direct_service_namespace,
                model_id_label,
                direct_port,
            )
        else:
            service_ip, _, gateway_port = find_gateway_endpoint(cmd, namespace, release)

        if not service_ip and context.dry_run:
            service_ip = "<dry-run-endpoint>"
            gateway_port = "80"

        return service_ip, gateway_port, is_standalone

    def run_health_checks(
        self,
        context: ExecutionContext,
        stack_path: Path,
    ) -> SmoketestReport:
        """Run the full health check suite: pods, /health, /v1/models,
        service endpoint, pod IPs, and OpenShift route.
        """
        report = SmoketestReport()
        # nok8s deploys plain containers: no Service, no pods, no route to
        # check. Probe the container endpoint over HTTP instead.
        if context.container_only:
            return nok8s_health_check(context, stack_path)
        cmd = context.require_cmd()
        namespace = context.require_namespace()
        plan_config = _load_config(stack_path)

        model_name = _nested_get(plan_config, "model", "name") or ""
        model_id_label = (
            plan_config.get("model_id_label", "")
            or _nested_get(plan_config, "model", "shortName")
            or ""
        )
        standalone_role = _nested_get(plan_config, "standalone", "role") or "standalone"
        is_kustomize = "kustomize" in context.deployed_methods
        guide_name = _nested_get(plan_config, "kustomize", "guideName") or ""

        service_ip, gateway_port, is_standalone = self.discover_endpoint(
            cmd,
            context,
            plan_config,
        )
        if (
            not is_standalone
            and not is_kustomize
            and _nested_get(plan_config, "gateway", "className") == "none"
        ):
            namespace = resolve_direct_service_namespace(plan_config, namespace)

        # 1. Check pods running for each configured role
        if is_kustomize:
            if guide_name in ("pd-disaggregation", "wide-ep"):
                roles_to_check = [("prefill", "prefill"), ("decode", "decode")]
            elif guide_name == "fast-model-actuation" or guide_name.startswith(
                "fast-model-actuation-"
            ):
                # Every FMA variant (base guide + derivatives like
                # fast-model-actuation-keda) has no decode/prefill Deployments:
                # launcher pods host vLLM and are intentionally sleeping (not all
                # Ready) and are not guide-labeled. The requester pods reserve
                # GPUs, carry the guide label -- so assert *those* are Ready.
                roles_to_check = [("requester", None)]
            else:
                roles_to_check = [("decode", "decode")]
        elif is_standalone:
            roles_to_check = [("standalone", standalone_role)]
        else:
            # Check whichever roles are configured (decode, prefill, or both)
            roles_to_check = []
            decode_enabled = _nested_get(plan_config, "decode", "enabled")
            decode_replicas = _nested_get(plan_config, "decode", "replicas") or 0
            # decode is enabled by default if not explicitly disabled
            if decode_enabled is not False and int(decode_replicas) > 0:
                roles_to_check.append(("decode", "decode"))
            else:
                context.logger.log_info(
                    "No decode pods configured -- skipping decode health check"
                )
            prefill_enabled = _nested_get(plan_config, "prefill", "enabled")
            prefill_replicas = _nested_get(plan_config, "prefill", "replicas") or 0
            if prefill_enabled and int(prefill_replicas) > 0:
                roles_to_check.append(("prefill", "prefill"))
            else:
                context.logger.log_info(
                    "No prefill pods configured -- skipping prefill health check"
                )

        if not roles_to_check:
            report.add(
                CheckResult(
                    "pods_configured",
                    False,
                    message="No decode, prefill, or standalone pods configured",
                )
            )

        for pod_type, role_label in roles_to_check:
            if is_kustomize and guide_name:
                role_selector = (
                    f"llm-d.ai/guide={guide_name.split('/')[-1]}"
                    if role_label is None
                    else f"llm-d.ai/guide={guide_name.split('/')[-1]},llm-d.ai/role={role_label}"
                )
            else:
                role_selector = (
                    f"llm-d.ai/model={model_id_label},llm-d.ai/role={role_label}"
                )
            context.logger.log_info(
                f"Checking {pod_type} pod status (selector: {role_selector})..."
            )
            pod_check = cmd.kube(
                "get",
                "pods",
                "-l",
                role_selector,
                "--namespace",
                namespace,
                "-o",
                "jsonpath={.items[*].status.phase}",
                check=False,
            )
            if not pod_check.dry_run:
                if pod_check.success:
                    phases = pod_check.stdout.strip().split()
                    if not phases:
                        report.add(
                            CheckResult(
                                f"{pod_type}_pods_exist",
                                False,
                                message=f"No {pod_type} pods found with selector '{role_selector}'",
                            )
                        )
                    elif not all(p == "Running" for p in phases):
                        report.add(
                            CheckResult(
                                f"{pod_type}_pods_running",
                                False,
                                expected="all Running",
                                actual=", ".join(phases),
                                message=f"Not all {pod_type} pods running (found: {', '.join(phases)})",
                            )
                        )
                    else:
                        context.logger.log_info(
                            f"All {len(phases)} {pod_type} pod(s) running (ok)"
                        )
                        report.add(
                            CheckResult(
                                f"{pod_type}_pods_running",
                                True,
                                message=f"{len(phases)} {pod_type} pod(s) running",
                            )
                        )
                        # Check pod Ready condition -- catches crash-looping
                        # sidecar containers (e.g., routing-proxy native
                        # sidecar with restartPolicy: Always). The Ready
                        # condition is True only when ALL containers pass
                        # their readiness probes.
                        ready_check = cmd.kube(
                            "get",
                            "pods",
                            "-l",
                            role_selector,
                            "--namespace",
                            namespace,
                            "--no-headers",
                            "-o",
                            "custom-columns=NAME:.metadata.name,READY:.status.containerStatuses[*].ready",
                            check=False,
                        )
                        if ready_check.success and ready_check.stdout.strip():
                            not_ready = []
                            for line in ready_check.stdout.strip().splitlines():
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    pod_name = parts[0]
                                    ready_values = parts[1]
                                    if "false" in ready_values.lower():
                                        not_ready.append(pod_name)
                                elif parts:
                                    not_ready.append(parts[0])
                            if not_ready:
                                report.add(
                                    CheckResult(
                                        f"{pod_type}_containers_ready",
                                        False,
                                        message=f"Not all {pod_type} pods ready (containers may be crash-looping): {', '.join(not_ready)}",
                                    )
                                )
                            else:
                                context.logger.log_info(
                                    f"All {pod_type} containers ready (ok)"
                                )
                else:
                    report.add(
                        CheckResult(
                            f"{pod_type}_pods_check",
                            False,
                            message=f"Failed to check {pod_type} pod status: {pod_check.stderr}",
                        )
                    )

        if not service_ip:
            report.add(
                CheckResult(
                    "endpoint_discovery",
                    False,
                    message="Could not find service/gateway IP",
                )
            )
            return report

        # When the scenario uses a shared HTTPRoute with path-based routing
        # (e.g. /pool-a/v1 -> pool A, /pool-b/v1 -> pool B), prepend this
        # stack's path prefix to the gateway URLs so requests actually hit
        # the InferencePool for THIS stack. Returns "" for the usual case
        # (single-model scenarios, standalone) - unchanged behavior.
        url_path_prefix = self._gateway_path_prefix_for_stack(
            plan_config,
            is_standalone,
            stack_name=stack_path.name,
        )

        # 2. Health check (/health) - skip when the endpoint doesn't
        # serve /health directly (EPP proxies /v1/* only; shared
        # HTTPRoute with rewriteTo narrows to /v1/* paths).
        if is_kustomize:
            context.logger.log_info(
                "Skipping /health probe: EPP proxies inference requests "
                "(/v1/*) only. /v1/models + direct-pod-IP health check "
                "still run."
            )
            report.add(
                CheckResult(
                    "health_endpoint_skipped",
                    True,
                    message="/health not served by EPP; relying on /v1/models + direct-pod-IP probe",
                )
            )
        elif url_path_prefix and not self._gateway_routes_health(plan_config):
            context.logger.log_info(
                f"Skipping /health probe: gateway rewriteTo="
                f"{(plan_config.get('httpRoute') or {}).get('rewriteTo', '/')!r} "
                "deliberately narrows routing to /v1/* paths. /v1/models "
                "probe + direct-pod-IP health check still run."
            )
            report.add(
                CheckResult(
                    "health_endpoint_skipped",
                    True,
                    message=(
                        "/health not routable via gateway (by design); "
                        "relying on /v1/models + direct-pod-IP probe"
                    ),
                )
            )
        else:
            health_err = self._check_health(
                cmd,
                context,
                namespace,
                service_ip,
                gateway_port,
                plan_config,
                url_path_prefix=url_path_prefix,
            )
            if health_err:
                report.add(CheckResult("health_endpoint", False, message=health_err))
            else:
                report.add(
                    CheckResult("health_endpoint", True, message="/health responding")
                )

        # 3. Wait for model ready (/v1/models)
        wait_err: str | None = None
        if report.passed:
            wait_err = self._wait_for_model_ready(
                cmd,
                context,
                namespace,
                service_ip,
                gateway_port,
                model_name,
                plan_config,
                url_path_prefix=url_path_prefix,
            )
            if wait_err:
                report.add(CheckResult("model_ready", False, message=wait_err))

        # 4. Test service/gateway. Skipped when model_ready already
        # timed out -- the assertion would fail with a cryptic Envoy
        # "upstream connection refused" that buries the real cause
        # (model still loading) under transport-layer noise.
        service_test_passed = False
        if wait_err:
            context.logger.log_info(
                "Skipping service/gateway assertion -- model_ready check "
                "already reported the underlying failure"
            )
            return report
        context.logger.log_info(
            f'Testing service/gateway "{service_ip}" (port {gateway_port})'
            f"{' [prefix=' + url_path_prefix + ']' if url_path_prefix else ''}..."
        )
        test_result = test_model_serving(
            cmd,
            namespace,
            service_ip,
            gateway_port,
            model_name,
            plan_config,
            max_retries=1,
            url_path_prefix=url_path_prefix,
        )
        if test_result:
            report.add(
                CheckResult(
                    "service_endpoint",
                    False,
                    message=f"Service test failed: {test_result}",
                )
            )
        else:
            service_test_passed = True
            context.logger.log_info(
                f"Service {service_ip}:{gateway_port} responding (ok)"
            )
            report.add(
                CheckResult("service_endpoint", True, message="Service responding")
            )

        # 5. Test pod IPs directly (use the first role -- decode for ms, standalone for standalone)
        inference_port = serving_port(plan_config)
        primary_role = ("decode", "decode")
        if roles_to_check:
            for role_info in roles_to_check:
                if role_info[0] in ("decode", "standalone"):
                    primary_role = role_info
                    break
            else:
                primary_role = roles_to_check[0]
        if primary_role[1] is None:
            # No role-labeled serving pod to probe directly (e.g. FMA: launcher pods
            # host vLLM but aren't guide-labeled, and the requester pods don't serve
            # inference). The service_endpoint check above already proved serving.
            context.logger.log_info(
                "No role-labeled serving pod -- skipping direct pod-IP probe"
            )
            pod_ips_result = None
        else:
            if is_kustomize and guide_name:
                primary_selector = f"llm-d.ai/guide={guide_name.split('/')[-1]},llm-d.ai/role={primary_role[1]}"
            else:
                primary_selector = (
                    f"llm-d.ai/model={model_id_label},llm-d.ai/role={primary_role[1]}"
                )
            pod_ips_result = cmd.kube(
                "get",
                "pods",
                "-l",
                primary_selector,
                "--namespace",
                namespace,
                "-o",
                "jsonpath={.items[*].status.podIP}",
                check=False,
            )

        if context.dry_run:
            test_model_serving(
                cmd,
                namespace,
                "<dry-run-pod-ip>",
                inference_port,
                model_name,
                plan_config,
                max_retries=1,
            )
        elif (
            pod_ips_result and pod_ips_result.success and pod_ips_result.stdout.strip()
        ):
            pod_ips = pod_ips_result.stdout.strip().split()
            for i, pod_ip in enumerate(pod_ips, 1):
                context.logger.log_info(
                    f"Testing pod {i}/{len(pod_ips)} at {pod_ip}:{inference_port}..."
                )
                test_result = test_model_serving(
                    cmd,
                    namespace,
                    pod_ip,
                    inference_port,
                    model_name,
                    plan_config,
                )
                if test_result:
                    if service_test_passed:
                        context.logger.log_warning(
                            f"Pod IP test failed (non-fatal, service "
                            f"test passed): {test_result}"
                        )
                    else:
                        report.add(
                            CheckResult(
                                f"pod_ip_{pod_ip}",
                                False,
                                message=f"Curl to {pod_ip}:{inference_port} failed: {test_result}",
                            )
                        )
                else:
                    context.logger.log_info(f"Pod {pod_ip} responding (ok)")

        # 6. OpenShift route (only for modelservice -- standalone/kustomize have no gateway route)
        if context.is_openshift and not is_standalone and not is_kustomize:
            context.logger.log_info("Testing OpenShift route...")
            self._test_openshift_route(
                cmd,
                context,
                namespace,
                model_name,
                plan_config,
                gateway_port,
                report,
                service_test_passed,
            )

        return report

    def run_inference_test(
        self,
        context: ExecutionContext,
        stack_path: Path,
    ) -> SmoketestReport:
        """Run a sample inference request and report pass/fail."""
        report = SmoketestReport()
        # nok8s has no ephemeral curl pod to exec from: POST directly.
        if context.container_only:
            return nok8s_inference_test(context, stack_path)
        cmd = context.require_cmd()
        namespace = context.require_namespace()
        plan_config = _load_config(stack_path)

        model_name = _nested_get(plan_config, "model", "name") or ""

        service_ip, gateway_port, _is_standalone = self.discover_endpoint(
            cmd,
            context,
            plan_config,
        )

        if not service_ip:
            report.add(
                CheckResult(
                    "inference_endpoint",
                    False,
                    message="Could not find service/gateway IP for inference test",
                )
            )
            return report

        protocol = "https" if str(gateway_port) == "443" else "http"
        # Shared-HTTPRoute scenarios route by path prefix (e.g. /pool-a/*
        # -> this stack's InferencePool). Bake the prefix into base_url so
        # every downstream {base_url}/v1/completions becomes
        # {base_url}/pool-a/v1/completions - the gateway then rewrites
        # /pool-a/* -> /* before the request reaches vLLM. Empty string for
        # every other scenario, preserving existing behavior.
        prefix = self._gateway_path_prefix_for_stack(
            plan_config,
            _is_standalone,
            stack_name=stack_path.name,
        )
        base_url = f"{protocol}://{service_ip}:{gateway_port}{prefix}"

        context.logger.log_info(f"Running sample inference against {base_url}...")

        # Try /v1/completions first
        context.logger.log_info("Trying /v1/completions endpoint...")
        result = self._try_completions(
            cmd,
            context,
            namespace,
            base_url,
            model_name,
            plan_config,
        )

        if result["success"]:
            self._print_demo_command(
                context,
                cmd,
                namespace,
                plan_config,
                base_url,
                "/v1/completions",
                result["payload"],
                result["generated_text"],
            )
            report.add(
                CheckResult(
                    "inference_completions",
                    True,
                    message=f'Inference passed via /v1/completions -- Generated: "{result["generated_text"]}"',
                )
            )
            return report

        # Fallback to /v1/chat/completions
        if result.get("should_fallback"):
            context.logger.log_info(
                f"/v1/completions returned non-transient error: "
                f"{result['error'][:100]}. Falling back to /v1/chat/completions..."
            )
            chat_result = self._try_chat_completions(
                cmd,
                context,
                namespace,
                base_url,
                model_name,
                plan_config,
            )
            if chat_result["success"]:
                self._print_demo_command(
                    context,
                    cmd,
                    namespace,
                    plan_config,
                    base_url,
                    "/v1/chat/completions",
                    chat_result["payload"],
                    chat_result["generated_text"],
                )
                report.add(
                    CheckResult(
                        "inference_chat",
                        True,
                        message=f'Inference passed via /v1/chat/completions -- Generated: "{chat_result["generated_text"]}"',
                    )
                )
                return report

            report.add(
                CheckResult(
                    "inference_test",
                    False,
                    message=(
                        f"/v1/completions failed: {result['error']}; "
                        f"/v1/chat/completions also failed: {chat_result['error']}"
                    ),
                )
            )
        else:
            report.add(
                CheckResult(
                    "inference_test",
                    False,
                    message=result.get("error", "Inference test failed"),
                )
            )

        return report

    def run_config_validation(
        self,
        context: ExecutionContext,
        stack_path: Path,
    ) -> SmoketestReport:
        """Validate deployed pod config matches scenario expectations.

        The base class returns an empty (all-pass) report.  Per-scenario
        validators override this to add scenario-specific checks.
        """
        report = SmoketestReport()
        report.add(
            CheckResult(
                "config_validation",
                True,
                message="No scenario-specific validator configured -- skipping config validation",
            )
        )
        return report

    @staticmethod
    def get_pod_specs(
        cmd: CommandExecutor,
        namespace: str,
        selector: str,
    ) -> list[dict]:
        """Fetch pod specs for all pods matching *selector*.

        Returns a list of pod dicts (items from ``kubectl get pods -o json``).
        """
        result = cmd.kube(
            "get",
            "pods",
            "-l",
            selector,
            "--namespace",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if not result.success or not result.stdout.strip():
            return []
        try:
            data = json.loads(result.stdout)
            return data.get("items", [])
        except json.JSONDecodeError:
            return []

    @staticmethod
    def get_pod_args(pod_spec: dict, container: str | None = None) -> str:
        """Extract the command/args string for a container (the engine's by default)."""
        container = container or _engine_container(pod_spec)
        for c in pod_spec.get("spec", {}).get("containers", []):
            if c.get("name") == container:
                args = c.get("args", [])
                return " ".join(args) if args else ""
        return ""

    @staticmethod
    def get_pod_image(pod_spec: dict, container: str | None = None) -> str:
        """Extract the image reference for a container (the engine's by default)."""
        container = container or _engine_container(pod_spec)
        for c in pod_spec.get("spec", {}).get("containers", []):
            if c.get("name") == container:
                return str(c.get("image", ""))
        return ""

    @staticmethod
    def get_pod_env(pod_spec: dict, container: str | None = None) -> dict[str, str]:
        """Extract env vars as a dict for a container (the engine's by default)."""
        container = container or _engine_container(pod_spec)
        for c in pod_spec.get("spec", {}).get("containers", []):
            if c.get("name") == container:
                return {
                    e["name"]: e.get("value", "")
                    for e in c.get("env", [])
                    if "name" in e
                }
        return {}

    @staticmethod
    def get_pod_resources(pod_spec: dict, container: str | None = None) -> dict:
        """Extract resources (limits + requests) for a container (the engine's by default)."""
        container = container or _engine_container(pod_spec)
        for c in pod_spec.get("spec", {}).get("containers", []):
            if c.get("name") == container:
                return c.get("resources", {})
        return {}

    @staticmethod
    def get_pod_containers(pod_spec: dict) -> list[str]:
        """Return names of all containers in the pod."""
        return [
            c.get("name", "") for c in pod_spec.get("spec", {}).get("containers", [])
        ]

    @staticmethod
    def get_pod_init_containers(pod_spec: dict) -> list[str]:
        """Return names of all init containers in the pod."""
        return [
            c.get("name", "")
            for c in pod_spec.get("spec", {}).get("initContainers", [])
        ]

    @staticmethod
    def get_pod_volumes(pod_spec: dict) -> list[str]:
        """Return names of all volumes in the pod."""
        return [v.get("name", "") for v in pod_spec.get("spec", {}).get("volumes", [])]

    @staticmethod
    def get_pod_annotations(pod_spec: dict) -> dict[str, str]:
        """Return annotations from the pod metadata."""
        return pod_spec.get("metadata", {}).get("annotations", {})

    @staticmethod
    def get_container_ports(pod_spec: dict, container: str | None = None) -> list[dict]:
        """Return container port entries for a container (the engine's by default)."""
        container = container or _engine_container(pod_spec)
        for c in pod_spec.get("spec", {}).get("containers", []):
            if c.get("name") == container:
                return c.get("ports", [])
        return []

    @staticmethod
    def assert_arg_present(pod_args: str, flag: str) -> CheckResult:
        """Check that *flag* appears in pod args (ignores value)."""
        if flag not in pod_args:
            return CheckResult(
                f"arg_{flag.lstrip('-')}",
                False,
                expected=f"{flag} present",
                actual="not found",
                message=f"{flag} not found in engine container args",
            )
        return CheckResult(
            f"arg_{flag.lstrip('-')}",
            True,
            message=f"{flag} present in engine container args",
        )

    @staticmethod
    def assert_arg_contains(
        pod_args: str,
        flag: str,
        value: str | None = None,
    ) -> CheckResult:
        """Check that *flag* (and optionally *value*) appears in pod args."""
        if flag not in pod_args:
            return CheckResult(
                f"arg_{flag.lstrip('-')}",
                False,
                expected=f"{flag} present",
                actual="not found",
                message=f"{flag} not found in engine container args",
            )
        if value is not None and value not in pod_args:
            return CheckResult(
                f"arg_{flag.lstrip('-')}",
                False,
                expected=f"{flag} with {value}",
                actual=f"{flag} present but value mismatch",
                message=f"{flag} found in engine container args but expected value '{value}' not present",
            )
        msg = (
            f"{flag} present"
            + (f" with {value}" if value else "")
            + " in engine container args"
        )
        return CheckResult(f"arg_{flag.lstrip('-')}", True, message=msg)

    @staticmethod
    def assert_arg_absent(pod_args: str, flag: str) -> CheckResult:
        """Check that *flag* does not appear in pod args."""
        if flag in pod_args:
            return CheckResult(
                f"no_{flag.lstrip('-')}",
                False,
                expected=f"{flag} absent",
                actual="present",
                message=f"{flag} should not be in engine container args",
            )
        return CheckResult(
            f"no_{flag.lstrip('-')}",
            True,
            message=f"{flag} correctly absent from engine container args",
        )

    @staticmethod
    def assert_env_equals(
        pod_env: dict[str, str],
        var_name: str,
        expected: str,
        container: str = ENGINE_CONTAINER,
        pod_name: str = "",
    ) -> CheckResult:
        """Check that env var *var_name* equals *expected* in the given container."""
        loc = (
            f"pod/{pod_name} container/{container}"
            if pod_name
            else f"container/{container}"
        )
        actual = pod_env.get(var_name)
        if actual is None:
            return CheckResult(
                f"env_{var_name}",
                False,
                expected=expected,
                actual="not set",
                message=f"{var_name} not set in {loc} env (expected {expected})",
            )
        if str(actual) != str(expected):
            return CheckResult(
                f"env_{var_name}",
                False,
                expected=expected,
                actual=str(actual),
                message=f"{var_name}={actual} in {loc} env (expected {expected})",
            )
        return CheckResult(
            f"env_{var_name}",
            True,
            message=f"{var_name}={actual} in {loc} env",
        )

    @staticmethod
    def expected_engine_repository(config: dict, *roles: str) -> str:
        """Image repository the first of *roles* with a resolved engine will run.

        ``<role>.engine.image`` is what resolve_engines filled in for whichever
        engine that role's command launches, so it is the right answer for a
        pod running SGLang as much as for one running vLLM. ``images.vllm`` is
        the last-resort fallback for a config that predates the per-role image.
        """
        for role in roles or ("decode", "standalone"):
            repo = _nested_get(config, role, "engine", "image", "repository")
            if repo:
                return str(repo)
        engine_name = _nested_get(config, "engine", "name") or "vllm"
        return str(_nested_get(config, "images", engine_name, "repository") or "")

    @staticmethod
    def _repository_of(image: str) -> str:
        """The repository out of an image reference.

        ``repo:tag``, ``repo@sha256:...`` and a bare ``repo`` all reduce to
        ``repo``. The tag is deliberately dropped: which build of an engine a
        pod runs is the image-pin tests' business, and a digest-pinned pod
        should still be recognised as that engine.
        """
        ref = str(image).split("@", 1)[0]
        head, sep, tail = ref.rpartition(":")
        # A colon in the last path segment is a tag; one before a `/` is a
        # registry port (`localhost:5000/vllm`), which is part of the repository.
        return head if sep and "/" not in tail else ref

    @classmethod
    def assert_engine_identity(
        cls,
        pod_spec: dict,
        engine_name: str,
        *,
        check_name: str = "engine",
        expected_repository: str = "",
        pod_name: str = "",
    ) -> CheckResult:
        """Check the pod is running the engine the plan says it is.

        ``<role>.engine.name`` is not evidence about a cluster: ``resolve_engines``
        wrote it by reading the scenario's own command, so a check that compares
        it against the plan compares the plan to itself and passes whatever is
        actually running. This asks the *pod*, and each fact is one the plan
        cannot supply:

          the launcher   ``detect_engine`` over the container's own args -- the
                         same signature matcher the resolver used, so no engine's
                         spelling is repeated here and a new ``EngineSpec`` is
                         covered by arriving.
          the image      the repository ``engine.imageKey`` resolved to. An
                         SGLang pod inspected against a TRT-LLM plan differs
                         here on the first token.
          the container  the serving container exists under the name every other
                         check inspects -- "flag not found" when the truth is
                         "container not found" is the one failure worse than
                         none.

        Two standups of one scenario in one namespace render the same Deployment
        name, so a pod from the *other* run answers this role's selector while
        every plan-versus-plan check still passes. That is what this catches.
        """
        spec = get_engine_spec(engine_name)
        # `spec.name` normalises an alias (`tensorrt-llm` -> `trtllm`) so the
        # comparison below is against the same spelling `detect_engine` returns.
        # A name no spec claims resolves to the generic spec, whose name would
        # otherwise be reported in place of the one the plan actually states.
        want = spec.name if is_known_engine(engine_name) else str(engine_name).strip()
        loc = f"pod/{pod_name} " if pod_name else ""
        containers = cls.get_pod_containers(pod_spec)
        serving = _engine_container(pod_spec)
        image = cls.get_pod_image(pod_spec)
        tokens, _ = tokenize(cls.get_pod_args(pod_spec))

        facts: list[str] = []
        problems: list[str] = []

        if serving in containers:
            facts.append(f"container '{serving}'")
        else:
            problems.append(
                f"no serving container (containers: [{', '.join(containers) or 'none'}])"
            )

        # A launcher signature is what there is to match, and two engines have
        # none: one this harness does not model at all, and `generic` -- the name
        # `resolve_engines` writes for a command whose launcher it did not
        # recognise, which is a supported way to run a scenario, not a fault. A
        # pod that carries its command in `command:` rather than `args:` likewise
        # gives us nothing to read. In each case the image below is the only
        # evidence available, and saying so beats inventing a pass.
        if not spec.launchers:
            facts.append(f"launcher unchecked (no launcher signature for '{want}')")
        elif not tokens:
            facts.append("launcher unchecked (no args on the container)")
        else:
            detected = detect_engine(tokens)
            if detected is None:
                problems.append(
                    f"args launch no engine this harness knows, not '{want}'"
                )
            elif detected.name != want:
                problems.append(f"args launch '{detected.name}', not '{want}'")
            else:
                facts.append("launcher in args")

        if expected_repository and image:
            got = cls._repository_of(image)
            # Suffix either way: a plan may name a repository the registry
            # prefixes (or the reverse) without that being a different engine.
            same = got == expected_repository or any(
                a.endswith("/" + b)
                for a, b in ((got, expected_repository), (expected_repository, got))
            )
            if same:
                facts.append(f"image {got}")
            else:
                problems.append(f"image is {got}, expected {expected_repository}")

        if problems:
            wanted = f"engine '{want}'"
            if expected_repository:
                wanted += f" from {expected_repository}"
            return CheckResult(
                check_name,
                False,
                expected=wanted,
                actual="; ".join(problems),
                message=f"{loc}is not running {wanted}: {'; '.join(problems)}",
            )
        return CheckResult(
            check_name,
            True,
            message=f"{loc}runs engine '{want}' ({', '.join(facts)})",
        )

    @staticmethod
    def _normalize_command(text: str) -> str:
        """Reduce a shell command to a comparable token stream.

        Backslash-newline continuations, indentation and the block folding the
        chart applies change the text without changing what runs, so compare
        tokens rather than characters.
        """
        return " ".join(text.replace("\\\n", " ").split())

    @classmethod
    def assert_command_rendered(
        cls,
        pod_args: str,
        expected_command: str,
        pod_name: str = "",
    ) -> CheckResult:
        """Check that the scenario's engine command reached the container intact.

        The rendered args are the preprocess step chained to the command, so
        containment (not equality) is the right relation. On a mismatch the
        message names the first token that diverges, because "the command does
        not match" on a 20-flag serve line is not an actionable sentence.
        """
        want = cls._normalize_command(expected_command)
        got = cls._normalize_command(pod_args)
        loc = f"pod/{pod_name} " if pod_name else ""
        flags = len([t for t in want.split() if t.startswith("-")])

        if want and want in got:
            return CheckResult(
                "engine_command",
                True,
                message=(
                    f"{loc}engine command rendered verbatim "
                    f"({len(want.split())} tokens, {flags} flags)"
                ),
            )

        # Name the divergence. Anchor on the command's first token (`vllm`,
        # `python3`, `trtllm-serve`, ...): everything before it in the args is
        # the preprocess step, which is not what we are comparing.
        want_tokens = want.split()
        head = want_tokens[0] if want_tokens else ""
        got_tokens = got.split()
        if head not in got_tokens:
            return CheckResult(
                "engine_command",
                False,
                expected=want,
                actual=got,
                message=(
                    f"{loc}engine command does not match the scenario: it does "
                    f"not start in the container args at all (no '{head}')"
                ),
            )
        tail = got_tokens[got_tokens.index(head) :]
        diff = "command absent from container args"
        for i, token in enumerate(want_tokens):
            if i >= len(tail):
                diff = f"truncated after '{' '.join(want_tokens[max(0, i - 1) : i]) or head}'"
                break
            if tail[i] != token:
                diff = f"expected '{token}', got '{tail[i]}'"
                break

        return CheckResult(
            "engine_command",
            False,
            expected=want,
            actual=got,
            message=f"{loc}engine command does not match the scenario: {diff}",
        )

    @staticmethod
    def assert_env_variant_list(
        pod_env: dict[str, str],
        var_name: str,
        expected_values: list,
        container: str = ENGINE_CONTAINER,
        pod_name: str = "",
    ) -> CheckResult:
        """Check a ``,,``-delimited per-replica env var.

        A scenario varies one value across replicas by writing a ``,,``-joined
        list in ``<role>.extraEnvVars`` and referencing the variable from its
        engine command::

            decode:
              extraEnvVars:
                - name: MY_MAX_LEN
                  value: "8192,,32768"
              engine:
                command: vllm serve Qwen/Qwen3-32B --max-model-len $MY_MAX_LEN

        The pod template is shared by all replicas, so the whole list is what
        lands in the pod spec; the preprocess script
        (`set_llmdbench_environment.py`) splits it at container start and
        re-exports the entry matching the pod's LWS index. Comparing the raw
        spec value against one scalar therefore always fails -- compare against
        the joined list instead.

        Only the list is verifiable from the pod spec. The per-index selection
        happens in the running container's shell and leaves no trace in the
        spec, so the resolved value is surfaced in the message for readability
        rather than asserted.
        """
        loc = (
            f"pod/{pod_name} container/{container}"
            if pod_name
            else f"container/{container}"
        )
        # A variant that omits the key renders as an empty segment: the
        # template maps the attribute without a default and this Jinja
        # environment is non-strict, so Undefined stringifies to "". Match
        # that instead of writing the literal "None".
        expected = ",,".join("" if v is None else str(v) for v in expected_values)
        actual = pod_env.get(var_name)
        detail = f"{len(expected_values)} variants, split per pod index at startup"
        if actual is None:
            return CheckResult(
                f"env_{var_name}",
                False,
                expected=expected,
                actual="not set",
                message=f"{var_name} not set in {loc} env (expected {expected})",
            )
        if str(actual) != expected:
            return CheckResult(
                f"env_{var_name}",
                False,
                expected=expected,
                actual=str(actual),
                message=(
                    f"{var_name}={actual} in {loc} env "
                    f"(expected {expected} -- {detail})"
                ),
            )
        return CheckResult(
            f"env_{var_name}",
            True,
            message=f"{var_name}={actual} in {loc} env ({detail})",
        )

    @staticmethod
    def assert_container_exists(containers: list[str], name: str) -> CheckResult:
        """Check that a container named *name* exists in the pod."""
        if name in containers:
            return CheckResult(
                f"container_{name}",
                True,
                message=f"Container '{name}' present in [{', '.join(containers)}]",
            )
        return CheckResult(
            f"container_{name}",
            False,
            expected=name,
            actual=str(containers),
            message=f"Container '{name}' not found in [{', '.join(containers)}]",
        )

    @staticmethod
    def assert_container_absent(containers: list[str], name: str) -> CheckResult:
        """Check that no container named *name* exists in the pod."""
        if name not in containers:
            return CheckResult(
                f"no_container_{name}",
                True,
                message=f"Container '{name}' correctly absent from [{', '.join(containers)}]",
            )
        return CheckResult(
            f"no_container_{name}",
            False,
            expected=f"no {name}",
            actual=f"{name} present",
            message=f"Container '{name}' should not be in [{', '.join(containers)}]",
        )

    @staticmethod
    def assert_replica_count(pods: list[dict], expected: int) -> CheckResult:
        """Check that the number of pods matches *expected*."""
        actual = len(pods)
        if actual == expected:
            return CheckResult(
                "replica_count",
                True,
                message=f"{actual} replica(s) (expected {expected})",
            )
        return CheckResult(
            "replica_count",
            False,
            expected=str(expected),
            actual=str(actual),
            message=f"{actual} replica(s) (expected {expected})",
        )

    @staticmethod
    def assert_resource_matches(
        actual_resources: dict,
        expected_value: str,
        resource_path: str,
    ) -> CheckResult:
        """Check a resource field like limits.memory or limits.nvidia.com/gpu.

        The path is ``<section>.<resourceName>`` and a Kubernetes extended
        resource name carries dots of its own (``nvidia.com/gpu``,
        ``habana.ai/gaudi``), so only the first dot separates the two.
        """
        section, _, field = resource_path.partition(".")
        val = (
            actual_resources.get(section)
            if isinstance(actual_resources, dict)
            else None
        )
        val = val.get(field) if isinstance(val, dict) else None

        if val is None:
            return CheckResult(
                f"resource_{resource_path}",
                False,
                expected=expected_value,
                actual="not set",
                message=f"engine container resources.{resource_path} not set (expected {expected_value})",
            )
        if str(val) != str(expected_value):
            return CheckResult(
                f"resource_{resource_path}",
                False,
                expected=expected_value,
                actual=str(val),
                message=f"engine container resources.{resource_path}={val} (expected {expected_value})",
            )
        return CheckResult(
            f"resource_{resource_path}",
            True,
            message=f"engine container resources.{resource_path}={val}",
        )

    def validate_role_pods(
        self,
        cmd: CommandExecutor,
        namespace: str,
        config: dict,
        role: str,
        model_short: str,
        report: SmoketestReport,
        logger=None,
        context: ExecutionContext | None = None,
    ) -> list[dict]:
        """Validate all aspects of pods for a given role (decode/prefill/standalone).

        Checks replica count, resources, parallelism, env vars, init containers,
        security context, volumes, probes, and engine args against the rendered config.

        Returns the list of matching pods.
        """
        role_config = _nested_get(config, role) or {}
        prefix = role  # used in check names

        # --- Replica count ---
        is_kustomize = (
            ("kustomize" in context.deployed_methods)
            if context
            else (_nested_get(config, "kustomize", "enabled") is True)
        )
        guide_name = _nested_get(config, "kustomize", "guideName") or ""

        if is_kustomize and guide_name:
            role_selector = (
                f"llm-d.ai/guide={guide_name.split('/')[-1]},llm-d.ai/role={role}"
            )
        else:
            role_selector = f"llm-d.ai/model={model_short},llm-d.ai/role={role}"

        pods = self.get_pod_specs(
            cmd,
            namespace,
            role_selector,
        )

        if not pods:
            return pods

        pod = pods[0]
        pod_name = pod.get("metadata", {}).get("name", "unknown")
        pod_node = pod.get("spec", {}).get("nodeName", "unknown")
        pod_ns = pod.get("metadata", {}).get("namespace", namespace)
        group_name = role

        # Emit a header check so the step renderer can group output
        report.add(
            CheckResult(
                name=f"{prefix}_header",
                passed=True,
                message=f"Inspecting {role} pod: {pod_name} (node: {pod_node}, ns: {pod_ns})",
                group=group_name,
                is_header=True,
            )
        )

        if is_kustomize:
            # Under kustomize, the deployment is defined entirely by the guide's
            # own manifests, ignoring the scenario's role-specific configs.
            # We return early here to only verify that the pods exist and run,
            # and skip the detailed config-matching checks.
            return pods

        expected_replicas = role_config.get("replicas")
        if expected_replicas is not None:
            expected_replicas = int(expected_replicas)
            # When multinode (LWS) is enabled, each replica spawns
            # ``workers`` pods (1 leader + N-1 workers).
            multinode_enabled = _nested_get(config, "multinode", "enabled")
            if multinode_enabled:
                workers = int(role_config.get("parallelism", {}).get("workers", 1))
                expected_pods = expected_replicas * workers
            else:
                expected_pods = expected_replicas
            pod_details = (
                ", ".join(
                    f"{p.get('metadata', {}).get('name', '?')}@{p.get('spec', {}).get('nodeName', '?')}"
                    for p in pods
                )
                or "none"
            )

            # If WVA + HPA owns the replica count for this role, the
            # actual pod count is HPA-driven and may legitimately differ
            # from the scenario's static `<role>.replicas` (e.g. with
            # `decode.replicas: 2` and `wva.hpa.minReplicas: 1`, an idle
            # cluster ends up at 1 pod, which is correct, not a regression).
            #
            # In that case we relax the check to "actual count is within
            # the HPA's [minReplicas, maxReplicas] window" and surface
            # both the scenario value and the HPA bounds in the message
            # so a reader can still tell what's happening.
            #
            # Multinode (LWS) deployments scale via LeaderWorkerSet, not
            # HPA, so the relaxation must not apply to them - keep strict
            # equality there.
            hpa_managed = (
                _nested_get(config, "wva", "enabled")
                and _nested_get(config, "wva", "hpa", "enabled")
                and role in _WVA_HPA_MANAGED_ROLES
                and not multinode_enabled
            )
            if hpa_managed:
                hpa_min = int(_nested_get(config, "wva", "hpa", "minReplicas") or 1)
                hpa_max = int(
                    _nested_get(config, "wva", "hpa", "maxReplicas") or expected_pods
                )
                in_range = hpa_min <= len(pods) <= hpa_max
                report.add(
                    CheckResult(
                        f"{prefix}_replicas",
                        in_range,
                        expected=f"{hpa_min}..{hpa_max} (HPA window)",
                        actual=str(len(pods)),
                        message=(
                            f"{role} pods in ns/{namespace}: "
                            f"{len(pods)} (HPA min={hpa_min} max={hpa_max}, "
                            f"scenario.{role}.replicas={expected_pods}) "
                            f"[{pod_details}]"
                        ),
                    )
                )
            else:
                report.add(
                    CheckResult(
                        f"{prefix}_replicas",
                        len(pods) == expected_pods,
                        expected=str(expected_pods),
                        actual=str(len(pods)),
                        message=(
                            f"{role} pods in ns/{namespace}: "
                            f"{len(pods)} (expected {expected_pods}) [{pod_details}]"
                        ),
                    )
                )

        def _tag(check: CheckResult) -> CheckResult:
            """Tag a CheckResult with its group for indented rendering."""
            check.group = group_name
            return check

        args = self.get_pod_args(pod)
        env = self.get_pod_env(pod)
        containers = self.get_pod_containers(pod)
        init_containers = self.get_pod_init_containers(pod)
        resources = self.get_pod_resources(pod)
        volumes = self.get_pod_volumes(pod)
        ports = self.get_container_ports(pod)

        # --- Resources (limits + requests) ---
        for section in ("limits", "requests"):
            for field in ("memory", "cpu", "ephemeral-storage"):
                expected = _nested_get(role_config, "resources", section, field)
                if expected is not None:
                    report.add(
                        _tag(
                            self.assert_resource_matches(
                                resources,
                                str(expected),
                                f"{section}.{field}",
                            )
                        )
                    )

        # --- Accelerators per pod, as the plan stated them ---
        # A device count is a Kubernetes fact, so it is resolved exactly the way
        # the renderer resolved it (`resources.limits.<resource>` ->
        # `accelerator.count` per role -> plan-wide -> `parallelism.tensor x
        # dataLocal`) and compared against what the pod actually got. A mismatch
        # means the chart and the plan disagree, which shows up at runtime as a
        # cryptic engine-side failure, so it is worth naming here. Zero means the
        # role is CPU-only and there is nothing to check.
        wanted_accelerators, _count_source = effective_accelerator_count(
            role_config, config
        )
        accel_resource = (
            _nested_get(config, "accelerator", "resource")
            or _nested_get(role_config, "accelerator", "resourceName")
            or _nested_get(config, "accelerator", "resourceName")
        )
        if wanted_accelerators and accel_resource:
            report.add(
                _tag(
                    self.assert_resource_matches(
                        resources,
                        str(wanted_accelerators),
                        f"limits.{accel_resource}",
                    )
                )
            )

        # --- Extra env vars ---
        extra_env = role_config.get("extraEnvVars", [])
        for ev in extra_env:
            ev_name = ev.get("name")
            ev_value = ev.get("value")
            if ev_name and ev_value is not None:
                report.add(
                    _tag(
                        self.assert_env_equals(
                            env, ev_name, str(ev_value), pod_name=pod_name
                        )
                    )
                )

        # --- Init containers ---
        expected_init = role_config.get("initContainers", [])
        for ic in expected_init:
            ic_name = ic.get("name")
            if ic_name:
                found = ic_name in init_containers
                report.add(
                    _tag(
                        CheckResult(
                            f"{prefix}_init_{ic_name}",
                            found,
                            expected=f"'{ic_name}' in initContainers",
                            actual=f"initContainers: [{', '.join(init_containers)}]",
                            message=f"initContainer '{ic_name}' {'present' if found else 'not found'} in [{', '.join(init_containers)}]",
                        )
                    )
                )

        # --- Security context capabilities ---
        extra_config = role_config.get("extraContainerConfig", {})
        expected_caps = _nested_get(
            extra_config, "securityContext", "capabilities", "add"
        )
        if expected_caps:
            actual_caps = self._get_container_security_caps(pod)
            for cap in expected_caps:
                has_cap = cap in actual_caps
                report.add(
                    _tag(
                        CheckResult(
                            f"{prefix}_cap_{cap}",
                            has_cap,
                            expected=cap,
                            actual=f"capabilities.add: [{', '.join(actual_caps)}]",
                            message=f"securityContext capability {cap} {'present' if has_cap else 'not found'} in [{', '.join(actual_caps)}]",
                        )
                    )
                )

        # --- Routing proxy (may be a regular container or init container) ---
        # The helm chart only injects the routing proxy on decode pods,
        # not prefill pods (prefill receives traffic via KV transfer).
        routing_enabled = _nested_get(config, "routing", "proxy", "enabled")
        if routing_enabled is True and role in ("decode", "standalone"):
            in_containers = "routing-proxy" in containers
            in_init = "routing-proxy" in init_containers
            found = in_containers or in_init
            location = (
                "containers" if in_containers else ("initContainers" if in_init else "")
            )
            report.add(
                _tag(
                    CheckResult(
                        f"{prefix}_routing_proxy",
                        found,
                        expected="routing-proxy in containers or initContainers",
                        actual=f"{'found in ' + location if found else 'not found'}",
                        message=f"routing-proxy {'present in ' + location if found else 'not found in containers or initContainers'}",
                    )
                )
            )
        elif routing_enabled is False and role in ("decode", "standalone"):
            in_containers = "routing-proxy" in containers
            in_init = "routing-proxy" in init_containers
            found = in_containers or in_init
            location = (
                "containers" if in_containers else ("initContainers" if in_init else "")
            )
            report.add(
                _tag(
                    CheckResult(
                        f"{prefix}_no_routing_proxy",
                        not found,
                        expected="routing-proxy absent",
                        actual=f"{'found in ' + location if found else 'absent'}",
                        message=f"routing-proxy {'should not be present but found in ' + location if found else 'correctly absent'}",
                    )
                )
            )

        # --- Volumes from the plan-wide `engine` section ---
        expected_volumes = _nested_get(config, "engine", "volumes") or []
        for vol in expected_volumes:
            vol_name = vol.get("name")
            if vol_name:
                found = vol_name in volumes
                report.add(
                    _tag(
                        CheckResult(
                            f"{prefix}_volume_{vol_name}",
                            found,
                            expected=f"'{vol_name}' in spec.volumes",
                            actual=f"spec.volumes: [{', '.join(volumes)}]",
                            message=f"volume '{vol_name}' {'present' if found else 'not found'} in spec.volumes [{', '.join(volumes)}]",
                        )
                    )
                )

        # --- Volume mounts from the plan-wide `engine` section ---
        expected_mounts = _nested_get(config, "engine", "volumeMounts") or []
        actual_mounts = self._get_container_volume_mounts(pod)
        for mount in expected_mounts:
            mount_name = mount.get("name")
            mount_path = mount.get("mountPath")
            if mount_name:
                has_mount = mount_name in actual_mounts
                actual_path = actual_mounts.get(mount_name, "N/A")
                mount_names = list(actual_mounts.keys())
                report.add(
                    _tag(
                        CheckResult(
                            f"{prefix}_mount_{mount_name}",
                            has_mount,
                            expected=f"'{mount_name}' at {mount_path}",
                            actual=f"{'at ' + actual_path if has_mount else 'not found'} in [{', '.join(mount_names)}]",
                            message=f"volumeMount '{mount_name}' {'at ' + actual_path if has_mount else 'not found in [' + ', '.join(mount_names) + ']'}",
                        )
                    )
                )

        # --- Probes ---
        probe_config = role_config.get("probes", {})
        self._validate_probes(pod, prefix, probe_config, report, group=group_name)

        # --- The engine command, verbatim ---
        # The scenario's command is the contract, and llm-d-benchmark's job is
        # to deliver it to the container unchanged. So assert exactly that. It
        # catches every way the chain between scenario and pod can damage a
        # command -- a dropped line continuation, a quote eaten by a shell
        # layer, ${...} substitution that left a placeholder behind, a role that
        # silently inherited the plan-wide command when it meant to override it
        # -- and it does so for vLLM, SGLang and TRT-LLM alike, with no
        # per-engine knowledge. Checking flag by flag instead would mean
        # re-deriving every engine's own spelling here, which is the bookkeeping
        # the verbatim contract exists to avoid.
        #
        # Whether the flags inside the command are the right flags is the
        # engine's judgement, not ours; a wrong one fails loudly at startup.
        expected_command = (
            _nested_get(role_config, "engine", "command")
            or _nested_get(config, "engine", "command")
            or ""
        )
        if str(expected_command).strip():
            report.add(
                _tag(
                    self.assert_command_rendered(
                        args, str(expected_command), pod_name=pod_name
                    )
                )
            )

        # --- The engine the pod is actually running ---
        # resolve_engines detected the engine from the scenario's command and
        # picked the image, health path and metrics path from it. Assert the
        # running pod agrees -- see `assert_engine_identity` for why the plan's
        # own `engine.name` is not evidence that it does.
        engine_cfg = role_config.get("engine") or {}
        engine_name = engine_cfg.get("name") or _nested_get(config, "engine", "name")
        if engine_name:
            report.add(
                _tag(
                    self.assert_engine_identity(
                        pod,
                        str(engine_name),
                        check_name=f"{prefix}_engine",
                        expected_repository=str(
                            _nested_get(role_config, "engine", "image", "repository")
                            or ""
                        ),
                        pod_name=pod_name,
                    )
                )
            )

        # --- The port the engine binds ---
        # Read out of the command by resolve_engines, then used for the
        # container port and the probes. A pod whose containerPort disagrees
        # with the command's --port passes its probes against nothing.
        engine_port = engine_cfg.get("port")
        if engine_port and ports:
            declared = [p.get("containerPort") for p in ports]
            has_port = int(engine_port) in [p for p in declared if p is not None]
            report.add(
                _tag(
                    CheckResult(
                        f"{prefix}_engine_port",
                        has_port,
                        expected=str(engine_port),
                        actual=", ".join(str(d) for d in declared),
                        message=(
                            f"engine port {engine_port} (from the command) "
                            f"{'present' if has_port else 'not found'} in "
                            f"containerPorts [{', '.join(str(d) for d in declared)}]"
                        ),
                    )
                )
            )

        return pods

    @staticmethod
    def _get_container_security_caps(
        pod_spec: dict,
        container: str | None = None,
    ) -> list[str]:
        """Extract security context capabilities for a container."""
        container = container or _engine_container(pod_spec)
        for c in pod_spec.get("spec", {}).get("containers", []):
            if c.get("name") == container:
                return (
                    c.get("securityContext", {}).get("capabilities", {}).get("add", [])
                )
        return []

    @staticmethod
    def _get_container_volume_mounts(
        pod_spec: dict,
        container: str | None = None,
    ) -> dict[str, str]:
        """Return volume mount names mapped to mount paths."""
        container = container or _engine_container(pod_spec)
        for c in pod_spec.get("spec", {}).get("containers", []):
            if c.get("name") == container:
                return {
                    m.get("name", ""): m.get("mountPath", "")
                    for m in c.get("volumeMounts", [])
                }
        return {}

    def _validate_probes(
        self,
        pod_spec: dict,
        prefix: str,
        probe_config: dict,
        report: SmoketestReport,
        container: str | None = None,
        group: str = "",
    ):
        """Validate probe configuration against config."""
        container = container or _engine_container(pod_spec)
        for c in pod_spec.get("spec", {}).get("containers", []):
            if c.get("name") != container:
                continue

            for probe_type in ("startup", "liveness", "readiness"):
                expected = probe_config.get(probe_type)
                if not expected:
                    continue

                actual_probe = c.get(f"{probe_type}Probe", {})
                if not actual_probe:
                    report.add(
                        CheckResult(
                            f"{prefix}_{probe_type}_probe",
                            False,
                            message=f"{probe_type}Probe not configured on container",
                            group=group,
                        )
                    )
                    continue

                # Check path
                expected_path = expected.get("path")
                if expected_path:
                    actual_path = actual_probe.get("httpGet", {}).get("path")
                    report.add(
                        CheckResult(
                            f"{prefix}_{probe_type}_path",
                            actual_path == expected_path,
                            expected=expected_path,
                            actual=str(actual_path),
                            message=f"{probe_type}Probe path: {actual_path} (expected {expected_path})",
                            group=group,
                        )
                    )

                # Check key numeric fields
                for field in (
                    "failureThreshold",
                    "periodSeconds",
                    "initialDelaySeconds",
                    "timeoutSeconds",
                ):
                    expected_val = expected.get(field)
                    if expected_val is not None:
                        actual_val = actual_probe.get(field)
                        report.add(
                            CheckResult(
                                f"{prefix}_{probe_type}_{field}",
                                str(actual_val) == str(expected_val),
                                expected=str(expected_val),
                                actual=str(actual_val),
                                message=f"{probe_type}Probe.{field}: {actual_val} (expected {expected_val})",
                                group=group,
                            )
                        )
            break

    def _check_health(
        self,
        cmd: CommandExecutor,
        context: ExecutionContext,
        namespace: str,
        host: str,
        port: str | int,
        plan_config: dict | None = None,
        timeout: int = 120,
        poll_interval: int = 10,
        url_path_prefix: str = "",
    ) -> str | None:
        # The engine and its health path come from the plan, where
        # `resolve_engines` published them for whichever launcher the scenario's
        # command names -- so this log line says "sglang" on an SGLang stack, and
        # an engine that serves its health elsewhere is polled where it serves.
        engine_cfg = serving_engine(plan_config or {})
        engine = str(engine_cfg.get("name") or "") or "the engine"
        health_path = str(engine_cfg.get("healthPath") or "/health")
        protocol = "https" if str(port) == "443" else "http"
        prefix = _normalize_url_prefix(url_path_prefix)
        url = f"{protocol}://{host}:{port}{prefix}{health_path}"
        curl_image = "quay.io/fedora/fedora"
        override_args = _build_overrides(plan_config)

        context.logger.log_info(
            f"Health check: verifying {engine} is listening at "
            f"{host}:{port}{prefix}{health_path}... Using curl image: {curl_image}"
        )
        start = time.time()
        attempt = 0

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                return (
                    f"{engine} health check failed: {health_path} did not "
                    f"respond after {timeout}s -- process may not be running"
                )

            attempt += 1
            pod_name = f"healthcheck-{_rand_suffix()}"
            curl_cmd = f"'curl -sk --max-time 10 -o /dev/null -w %{{http_code}} {url}'"

            kubectl_args = (
                [
                    "run",
                    pod_name,
                    "--rm",
                    "--attach",
                    "--quiet",
                    "--restart=Never",
                    "--namespace",
                    namespace,
                    f"--image={curl_image}",
                ]
                + _ephemeral_label_args()
                + override_args
                + ["--command", "--", "sh", "-c", curl_cmd]
            )
            result = cmd.kube(*kubectl_args, check=False)

            if result.dry_run:
                return None

            status_code = result.stdout.strip() if result.success else ""

            if status_code == "200":
                context.logger.log_info(
                    f"{engine} health check passed (ok) ({int(elapsed)}s elapsed)"
                )
                return None

            remaining = int(timeout - elapsed)
            context.logger.log_info(
                f"{engine} not listening yet (attempt {attempt}, "
                f"status={status_code or 'N/A'}, {remaining}s remaining)..."
            )
            time.sleep(poll_interval)

    # Default 30 minutes -- accommodates large models (DeepSeek-R1,
    # Llama-3.1-405B, Qwen3-235B etc.) where weight download + load
    # across N workers can take 15+ minutes. Small models finish in
    # seconds and exit immediately; the larger ceiling is upside-free.
    _DEFAULT_MODEL_READY_TIMEOUT = 1800
    _DEFAULT_MODEL_READY_POLL_INTERVAL = 15

    def _wait_for_model_ready(
        self,
        cmd: CommandExecutor,
        context: ExecutionContext,
        namespace: str,
        host: str,
        port: str | int,
        expected_model: str,
        plan_config: dict | None = None,
        timeout: int | None = None,
        poll_interval: int | None = None,
        url_path_prefix: str = "",
    ) -> str | None:
        """Poll ``/v1/models`` until the expected model is served.

        Returns ``None`` on success, or a human-readable error message
        on timeout (caller is expected to surface it as a check failure
        -- the historical "log warning and silently proceed" was wrong
        because the subsequent assertion would then fail with the
        cryptic Envoy "upstream connection refused" instead of the
        actual cause).

        Timeout / poll interval can be overridden per-scenario via
        ``harness.smoketest.modelReadyTimeout`` and
        ``harness.smoketest.modelReadyPollInterval`` in the plan
        config; explicit kwargs win over plan-config values which win
        over the class defaults.
        """
        timeout = self._resolve_wait_setting(
            plan_config,
            "modelReadyTimeout",
            timeout,
            self._DEFAULT_MODEL_READY_TIMEOUT,
        )
        poll_interval = self._resolve_wait_setting(
            plan_config,
            "modelReadyPollInterval",
            poll_interval,
            self._DEFAULT_MODEL_READY_POLL_INTERVAL,
        )

        context.logger.log_info(
            f"Waiting for model '{expected_model}' at {host}:{port} "
            f"(timeout {timeout}s, poll every {poll_interval}s)..."
        )
        start = time.time()
        attempt = 0
        last_progress_log = 0.0
        # Log first 3 polls verbosely, then once per ~60s. A 30-minute
        # wait at 15s polls would otherwise produce 120 "not ready yet"
        # lines -- noise that obscures the surrounding standup output.
        verbose_attempts = 3
        progress_interval = 60.0

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                err = (
                    f"Model '{expected_model}' did not become ready at "
                    f"{host}:{port} within {timeout}s. For large models "
                    f"(DeepSeek-R1, Llama-3.1-405B, etc.) increase the "
                    f"wait via `harness.smoketest.modelReadyTimeout: 3600` "
                    f"in your scenario, or check the model-server logs:\n"
                    f"  kubectl logs -n {namespace} "
                    f"-l llm-d.ai/role=decode -c {ENGINE_CONTAINER} --tail=100"
                )
                context.logger.log_error(err)
                return err

            attempt += 1
            result = test_model_serving(
                cmd,
                namespace,
                host,
                port,
                expected_model,
                plan_config,
                max_retries=1,
                url_path_prefix=url_path_prefix,
            )

            if cmd.dry_run:
                return None

            if result is None:
                context.logger.log_info(
                    f"Model '{expected_model}' ready at {host}:{port} "
                    f"({int(elapsed)}s, attempt {attempt})"
                )
                return None

            remaining = int(timeout - elapsed)
            if attempt <= verbose_attempts or (
                elapsed - last_progress_log >= progress_interval
            ):
                context.logger.log_info(
                    f"Model not ready yet (attempt {attempt}, "
                    f"{int(elapsed)}s elapsed, {remaining}s remaining)..."
                )
                last_progress_log = elapsed
            time.sleep(poll_interval)

    @staticmethod
    def _resolve_wait_setting(
        plan_config: dict | None,
        key: str,
        explicit: int | None,
        default: int,
    ) -> int:
        """Pick a wait timeout / poll interval.

        Priority: explicit kwarg > plan_config.harness.smoketest.<key> >
        class default. Plan-config values that aren't a positive int are
        ignored (with no log -- scenario authors typo'ing this is rare
        enough that we'd rather not fire false-positive warnings).
        """
        if explicit is not None:
            return explicit
        if plan_config:
            cfg = ((plan_config.get("harness") or {}).get("smoketest") or {}).get(key)
            if cfg is not None:
                try:
                    cfg_int = int(cfg)
                    if cfg_int > 0:
                        return cfg_int
                except (TypeError, ValueError):
                    pass
        return default

    def _test_openshift_route(
        self,
        cmd: CommandExecutor,
        context: ExecutionContext,
        namespace: str,
        model_name: str,
        plan_config: dict,
        gateway_port: str,
        report: SmoketestReport,
        service_test_passed: bool,
    ):
        # epponly/none deploy no Gateway, so no route exists to fetch --
        # looking one up there is a guaranteed warning on every run.
        gateway_class = _nested_get(plan_config, "gateway", "className") or ""
        if gateway_class in ("epponly", "none"):
            context.logger.log_info(
                f"gateway.className={gateway_class} -- no OpenShift route to test"
            )
            return

        release = _nested_get(plan_config, "release") or ""
        route_name = f"{release}-inference-gateway-route"

        route_result = cmd.kube(
            "get",
            "route",
            route_name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.spec.host}:{.spec.tls.termination}",
            check=False,
        )
        if route_result.success and route_result.stdout.strip():
            parts = route_result.stdout.strip().strip("'").split(":", 1)
            route_host = parts[0]
            tls_termination = parts[1] if len(parts) > 1 else ""
            route_port = "443" if tls_termination else "80"

            context.logger.log_info(
                f"Testing route {route_host} (port {route_port})..."
            )
            test_result = test_model_serving(
                cmd,
                namespace,
                route_host,
                route_port,
                model_name,
                plan_config,
            )
            if test_result:
                if service_test_passed:
                    context.logger.log_warning(
                        f"Route test failed (non-fatal): {test_result}"
                    )
                else:
                    report.add(
                        CheckResult(
                            "openshift_route",
                            False,
                            message=f"Route test failed: {test_result}",
                        )
                    )
            else:
                context.logger.log_info(f"Route {route_host} responding (ok)")
                report.add(
                    CheckResult(
                        "openshift_route",
                        True,
                        message="Route responding",
                    )
                )
        else:
            context.logger.log_warning(
                f"Unable to fetch OpenShift route '{route_name}'"
            )

    # Inference retry budget. Sequence with default values:
    #   attempt 1 -> wait 15s -> attempt 2 -> wait 30s -> attempt 3
    #   -> wait 60s -> attempt 4 -> wait 120s -> attempt 5 -> fallback
    # Total wait budget: ~225s (~3.75 min) covering most warmup races.
    # Tunable per-scenario via:
    #   harness.smoketest.inferenceMaxRetries
    #   harness.smoketest.inferenceRetryBaseInterval
    #   harness.smoketest.inferenceRetryMaxInterval
    _DEFAULT_INFERENCE_MAX_RETRIES = 5
    _DEFAULT_INFERENCE_RETRY_BASE = 15
    _DEFAULT_INFERENCE_RETRY_MAX = 120

    @staticmethod
    def _compute_backoff(attempt: int, base: int, cap: int) -> int:
        """Exponential backoff: ``base * 2 ** (attempt - 1)``, capped.

        Attempt is 1-indexed (so attempt 1 -> base, attempt 2 -> 2*base,
        etc). The cap prevents the wait from running unbounded if a
        future scenario sets a high max_retries.
        """
        if attempt < 1:
            return base
        return min(cap, base * (2 ** (attempt - 1)))

    def _try_completions(
        self,
        cmd: CommandExecutor,
        context: ExecutionContext,
        namespace: str,
        base_url: str,
        model_name: str,
        plan_config: dict | None,
        max_retries: int | None = None,
        retry_interval: int | None = None,
        retry_max_interval: int | None = None,
    ) -> dict:
        url = f"{base_url}/v1/completions"
        payload = {
            "model": model_name,
            "prompt": "The capital of the United States is",
            "max_tokens": 5,
            "temperature": 0,
        }

        # Resolve retry budget from plan config (scenarios can tune)
        # falling back to the class defaults.
        max_retries = self._resolve_wait_setting(
            plan_config,
            "inferenceMaxRetries",
            max_retries,
            self._DEFAULT_INFERENCE_MAX_RETRIES,
        )
        retry_interval = self._resolve_wait_setting(
            plan_config,
            "inferenceRetryBaseInterval",
            retry_interval,
            self._DEFAULT_INFERENCE_RETRY_BASE,
        )
        retry_max_interval = self._resolve_wait_setting(
            plan_config,
            "inferenceRetryMaxInterval",
            retry_max_interval,
            self._DEFAULT_INFERENCE_RETRY_MAX,
        )

        for attempt in range(1, max_retries + 1):
            stdout, err = self._curl_post(cmd, namespace, url, payload, plan_config)

            if cmd.dry_run:
                return {
                    "success": True,
                    "payload": payload,
                    "generated_text": "<dry-run>",
                }

            if err:
                if _is_retryable(err) and attempt < max_retries:
                    context.logger.log_info(
                        f"Attempt {attempt}/{max_retries}: {err[:80]}, "
                        f"retrying in {self._compute_backoff(attempt, retry_interval, retry_max_interval)}s..."
                    )
                    time.sleep(
                        self._compute_backoff(
                            attempt, retry_interval, retry_max_interval
                        )
                    )
                    continue
                return {"success": False, "error": err}

            try:
                resp = json.loads(stdout)
            except json.JSONDecodeError:
                # JSON decode failed -- empty body, partial body, or
                # garbage. All three are dominated by transient
                # warmup races (model server still loading after
                # /v1/models began responding, gateway flapping
                # mid-request, sim's request handler not yet bound).
                # We have no negative signal -- the HTTP status was
                # already vetted as 2xx by _curl_post -- so retry
                # unconditionally up to max_retries before falling
                # back to /v1/chat/completions. Don't gate on
                # _is_retryable here: that function looks for known
                # error strings (502/503/504/...) but for "no body"
                # there's nothing to match, and the historical
                # `_is_retryable("") -> False` made every transient
                # warmup race a hard smoketest failure.
                body_preview = stdout[:200].strip() or "(empty body)"
                if attempt < max_retries:
                    context.logger.log_info(
                        f"Attempt {attempt}/{max_retries}: non-JSON body "
                        f"({body_preview[:60]}), retrying in {retry_interval}s..."
                    )
                    time.sleep(
                        self._compute_backoff(
                            attempt, retry_interval, retry_max_interval
                        )
                    )
                    continue
                # Out of retries -- fall back to /v1/chat/completions.
                # Modern chat-only model servers and the llm-d
                # simulator both produce empty /v1/completions
                # responses; the chat endpoint usually works.
                # Chat-completions has no further fallback target so
                # its should_fallback is a no-op (kept for symmetric
                # error shape).
                return {
                    "success": False,
                    "error": f"Non-JSON response from {url}: {body_preview}",
                    "should_fallback": True,
                }

            if _is_non_transient_error(resp):
                error_msg = resp["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                return {
                    "success": False,
                    "error": str(error_msg),
                    "should_fallback": True,
                }

            if "error" in resp:
                error_msg = resp["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                if attempt < max_retries:
                    time.sleep(
                        self._compute_backoff(
                            attempt, retry_interval, retry_max_interval
                        )
                    )
                    continue
                return {"success": False, "error": str(error_msg)}

            if "choices" not in resp or not resp["choices"]:
                if attempt < max_retries:
                    time.sleep(
                        self._compute_backoff(
                            attempt, retry_interval, retry_max_interval
                        )
                    )
                    continue
                return {
                    "success": False,
                    "error": f"Missing choices in response from {url}",
                }

            first = resp["choices"][0]
            if not first.get("text") and not first.get("message"):
                if attempt < max_retries:
                    time.sleep(
                        self._compute_backoff(
                            attempt, retry_interval, retry_max_interval
                        )
                    )
                    continue
                return {"success": False, "error": f"No generated text from {url}"}

            text = first.get("text", "").strip()
            return {"success": True, "generated_text": text, "payload": payload}

        return {"success": False, "error": f"Exhausted {max_retries} retries for {url}"}

    def _try_chat_completions(
        self,
        cmd: CommandExecutor,
        context: ExecutionContext,
        namespace: str,
        base_url: str,
        model_name: str,
        plan_config: dict | None,
        max_retries: int | None = None,
        retry_interval: int | None = None,
        retry_max_interval: int | None = None,
    ) -> dict:
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "What is the capital of the United States?"}
            ],
            "max_tokens": 5,
            "temperature": 0,
        }

        # Same retry budget as /v1/completions -- the chat fallback also
        # benefits from giving the server time to warm up.
        max_retries = self._resolve_wait_setting(
            plan_config,
            "inferenceMaxRetries",
            max_retries,
            self._DEFAULT_INFERENCE_MAX_RETRIES,
        )
        retry_interval = self._resolve_wait_setting(
            plan_config,
            "inferenceRetryBaseInterval",
            retry_interval,
            self._DEFAULT_INFERENCE_RETRY_BASE,
        )
        retry_max_interval = self._resolve_wait_setting(
            plan_config,
            "inferenceRetryMaxInterval",
            retry_max_interval,
            self._DEFAULT_INFERENCE_RETRY_MAX,
        )

        for attempt in range(1, max_retries + 1):
            stdout, err = self._curl_post(cmd, namespace, url, payload, plan_config)

            if err:
                if _is_retryable(err) and attempt < max_retries:
                    context.logger.log_info(
                        f"Chat attempt {attempt}/{max_retries}: {err[:80]}, "
                        f"retrying in {self._compute_backoff(attempt, retry_interval, retry_max_interval)}s..."
                    )
                    time.sleep(
                        self._compute_backoff(
                            attempt, retry_interval, retry_max_interval
                        )
                    )
                    continue
                return {"success": False, "error": err}

            try:
                resp = json.loads(stdout)
            except json.JSONDecodeError:
                # JSON decode failed -- empty body, partial body, or
                # garbage. All three are dominated by transient
                # warmup races (model server still loading after
                # /v1/models began responding, gateway flapping
                # mid-request, sim's request handler not yet bound).
                # We have no negative signal -- the HTTP status was
                # already vetted as 2xx by _curl_post -- so retry
                # unconditionally up to max_retries before falling
                # back to /v1/chat/completions. Don't gate on
                # _is_retryable here: that function looks for known
                # error strings (502/503/504/...) but for "no body"
                # there's nothing to match, and the historical
                # `_is_retryable("") -> False` made every transient
                # warmup race a hard smoketest failure.
                body_preview = stdout[:200].strip() or "(empty body)"
                if attempt < max_retries:
                    context.logger.log_info(
                        f"Attempt {attempt}/{max_retries}: non-JSON body "
                        f"({body_preview[:60]}), retrying in {retry_interval}s..."
                    )
                    time.sleep(
                        self._compute_backoff(
                            attempt, retry_interval, retry_max_interval
                        )
                    )
                    continue
                # Out of retries -- fall back to /v1/chat/completions.
                # Modern chat-only model servers and the llm-d
                # simulator both produce empty /v1/completions
                # responses; the chat endpoint usually works.
                # Chat-completions has no further fallback target so
                # its should_fallback is a no-op (kept for symmetric
                # error shape).
                return {
                    "success": False,
                    "error": f"Non-JSON response from {url}: {body_preview}",
                    "should_fallback": True,
                }

            if "error" in resp:
                error_msg = resp["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                if _is_retryable(str(error_msg)) and attempt < max_retries:
                    time.sleep(
                        self._compute_backoff(
                            attempt, retry_interval, retry_max_interval
                        )
                    )
                    continue
                return {"success": False, "error": str(error_msg)}

            if "choices" not in resp or not resp["choices"]:
                if attempt < max_retries:
                    time.sleep(
                        self._compute_backoff(
                            attempt, retry_interval, retry_max_interval
                        )
                    )
                    continue
                return {"success": False, "error": f"Missing choices from {url}"}

            message = resp["choices"][0].get("message", {})
            text = message.get("content", "").strip()
            if not text:
                if attempt < max_retries:
                    time.sleep(
                        self._compute_backoff(
                            attempt, retry_interval, retry_max_interval
                        )
                    )
                    continue
                return {"success": False, "error": f"No content in response from {url}"}

            return {"success": True, "generated_text": text, "payload": payload}

        return {"success": False, "error": f"Exhausted {max_retries} retries for {url}"}

    @staticmethod
    def _curl_post(
        cmd: CommandExecutor,
        namespace: str,
        url: str,
        payload: dict,
        plan_config: dict | None,
        timeout_seconds: int = 120,
    ) -> tuple[str, str | None]:
        override_args = _build_overrides(plan_config)
        curl_image = "quay.io/fedora/fedora"
        pod_name = f"inference-test-{_rand_suffix()}"
        payload_json = json.dumps(payload)
        payload_b64 = base64.b64encode(payload_json.encode()).decode()

        # `-w '\n%{http_code}'` appends a trailing line with the HTTP
        # status so an empty body is still diagnosable (empty + 404 vs
        # empty + 502 vs empty + 200 look identical to `-s` alone).
        # We split the status off in Python before returning the body.
        curl_cmd = (
            f"'echo {payload_b64} | base64 -d | "
            f"curl -sk --max-time {timeout_seconds} "
            f'-w "\\n%{{http_code}}" '
            f"-X POST {url} "
            f'-H "Content-Type: application/json" '
            f"-d @- 2>&1'"
        )

        kubectl_args = (
            [
                "run",
                pod_name,
                "--rm",
                "--attach",
                "--quiet",
                "--restart=Never",
                "--namespace",
                namespace,
                f"--image={curl_image}",
            ]
            + _ephemeral_label_args()
            + override_args
            + ["--command", "--", "sh", "-c", curl_cmd]
        )

        result = cmd.kube(*kubectl_args, check=False)

        if result.dry_run:
            return "", None

        if not result.success:
            detail = result.stderr[:300] or result.stdout[:300]
            return "", f"Curl to {url} failed: {detail}"

        # Split the trailing HTTP status off from the body. Curl writes
        # the body followed by "\n<status>". A non-2xx status with an
        # empty body becomes a meaningful error string instead of
        # disappearing into "Non-JSON response: (empty body)".
        stdout = result.stdout
        body, _, status_part = stdout.rpartition("\n")
        status = status_part.strip()
        body = body.strip()
        if status and not status.startswith("2"):
            # Body of error responses (when the server bothered to send
            # one) is more useful than the status alone, so include both.
            body_preview = body[:200] or "(empty body)"
            return body, (f"Curl POST {url} returned HTTP {status}: {body_preview}")
        return body, None

    def _print_demo_command(
        self,
        context: ExecutionContext,
        cmd: CommandExecutor,
        namespace: str,
        plan_config: dict,
        base_url: str,
        endpoint: str,
        payload: dict,
        generated_text: str,
    ):
        payload_compact = json.dumps(payload, separators=(",", ":"))  # noqa: F841

        context.logger.log_info(f"✅ Inference test passed via {endpoint}")
        if generated_text:
            context.logger.log_info(f'   Generated: "{generated_text[:80]}"')
        context.logger.log_info("")

        external_url = self._detect_external_url(
            cmd,
            namespace,
            plan_config,
            endpoint,
        )

        demo_url = external_url or f"{base_url}{endpoint}"
        payload_pretty = json.dumps(payload, indent=2)
        context.logger.log_info("   To reproduce or demo, run:")
        context.logger.log_info("")
        context.logger.log_info("   curl -sk -X POST \\")
        context.logger.log_info(f"     {demo_url} \\")
        context.logger.log_info("     -H 'Content-Type: application/json' \\")
        context.logger.log_info("     -d '{")
        for line in payload_pretty.splitlines()[1:]:
            context.logger.log_info(f"       {line}")
        context.logger.log_info("     '")

    def _detect_external_url(
        self,
        cmd: CommandExecutor,
        namespace: str,
        plan_config: dict,
        endpoint: str,
    ) -> str | None:
        try:
            release = _nested_get(plan_config, "release") or ""
            model_id_label = (
                plan_config.get("model_id_label", "")
                or _nested_get(plan_config, "model", "shortName")
                or ""
            )
        except KeyError:
            return None

        if not release:
            return None

        route_name = f"{release}-inference-gateway-route"
        result = cmd.kube(
            "get",
            "route",
            route_name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.spec.host}:{.spec.tls.termination}",
            check=False,
        )

        if not result.success or not result.stdout.strip():
            return None

        parts = result.stdout.strip().strip("'").split(":", 1)
        route_host = parts[0]
        tls_termination = parts[1] if len(parts) > 1 else ""
        protocol = "https" if tls_termination else "http"

        return f"{protocol}://{route_host}/{model_id_label}{endpoint}"


def _nested_get(d: dict, *keys: str):
    """Safely traverse nested dicts."""
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
        if d is None:
            return None
    return d


def _load_config(stack_path: Path) -> dict:
    """Load the rendered config.yaml from a stack directory."""
    import yaml

    config_file = stack_path / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    return {}
