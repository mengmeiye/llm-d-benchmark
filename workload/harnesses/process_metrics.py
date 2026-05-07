#!/usr/bin/env python3

# Copyright 2025 The llm-d Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Metrics processing script for llm-d-benchmark
Parses and aggregates Prometheus metrics and vLLM logs
"""

import os
import re
import json
import glob
from collections import defaultdict
import statistics

metrics_dir = os.environ.get('METRICS_DIR', 'metrics')
raw_dir = os.path.join(metrics_dir, 'raw')
processed_dir = os.path.join(metrics_dir, 'processed')

# Metrics to aggregate across all pods for cluster-wide stats
AGGREGATE_METRICS = {
    'vllm:kv_cache_usage_perc', 'vllm:num_requests_running',
    'vllm:num_requests_waiting', 'vllm:num_preemptions_total',
    'vllm:prefix_cache_hit_rate', 'vllm:external_prefix_cache_hit_rate',
    # EPP pool-level gauges (already aggregated by EPP across endpoints)
    'inference_pool_average_kv_cache_utilization',
    'inference_pool_average_queue_size',
    'inference_pool_average_running_requests',
    'inference_pool_ready_pods',
}

# Ratio metrics: (output_name, numerator_metric, denominator_metric)
RATIO_METRICS = [
    ('vllm:prefix_cache_hit_rate',
     'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_queries_total'),
    ('vllm:external_prefix_cache_hit_rate',
     'vllm:external_prefix_cache_hits_total',
     'vllm:external_prefix_cache_queries_total'),
]

# Metric name -> unit mapping
METRIC_UNITS = {
    # Cache metrics
    'vllm:kv_cache_usage_perc': '%',
    'vllm:gpu_cache_usage_perc': '%',
    'vllm:cpu_cache_usage_perc': '%',
    'vllm:prefix_cache_hits_total': 'tokens',
    'vllm:prefix_cache_queries_total': 'tokens',
    'vllm:external_prefix_cache_hits_total': 'tokens',
    'vllm:external_prefix_cache_queries_total': 'tokens',
    # Memory metrics
    'vllm:gpu_memory_usage_bytes': 'bytes',
    'DCGM_FI_DEV_FB_USED': 'bytes',
    'vllm:cpu_memory_usage_bytes': 'bytes',
    'container_memory_usage_bytes': 'bytes',
    # Compute metrics
    'DCGM_FI_DEV_GPU_UTIL': '%',
    'container_cpu_usage_seconds_total': 'seconds',
    # Performance metrics
    'DCGM_FI_DEV_POWER_USAGE': 'watts',
    # NIXL KV transfer metrics
    'vllm:nixl_xfer_time_seconds_sum': 'seconds',
    'vllm:nixl_xfer_time_seconds_count': 'count',
    'vllm:nixl_bytes_transferred_sum': 'bytes',
    'vllm:nixl_bytes_transferred_count': 'count',
    # Preemption metrics
    'vllm:num_preemptions_total': 'count',
    # Queue metrics
    'vllm:num_requests_running': 'count',
    'vllm:num_requests_waiting': 'count',
    'vllm:num_requests_swapped': 'count',
    # Computed ratio metrics
    'vllm:prefix_cache_hit_rate': '%',
    'vllm:external_prefix_cache_hit_rate': '%',
    # EPP (inference scheduler) Prometheus metrics
    'inference_pool_average_kv_cache_utilization': '%',
    'inference_pool_average_queue_size': 'count',
    'inference_pool_average_running_requests': 'count',
    'inference_pool_ready_pods': 'count',
    'inference_extension_scheduler_e2e_duration_seconds_bucket': 'seconds',
    'inference_extension_scheduler_e2e_duration_seconds_sum': 'seconds',
    'inference_extension_scheduler_e2e_duration_seconds_count': 'count',
    'inference_extension_plugin_duration_seconds_bucket': 'seconds',
    'inference_extension_plugin_duration_seconds_sum': 'seconds',
    'inference_extension_plugin_duration_seconds_count': 'count',
    'inference_extension_request_duration_seconds_bucket': 'seconds',
    'inference_extension_request_duration_seconds_sum': 'seconds',
    'inference_extension_request_duration_seconds_count': 'count',
    'inference_extension_request_ttft_duration_seconds_bucket': 'seconds',
    'inference_extension_request_ttft_duration_seconds_sum': 'seconds',
    'inference_extension_request_ttft_duration_seconds_count': 'count',
    'inference_extension_input_tokens_bucket': 'tokens',
    'inference_extension_output_tokens_bucket': 'tokens',
    'inference_extension_normalized_time_per_output_token_bucket': 'seconds',
    'inference_extension_prefix_indexer_hit_ratio_bucket': 'ratio',
    'inference_extension_prefix_indexer_size': 'count',
    'llm_d_inference_scheduler_pd_decision_total': 'count',
    'llm_d_inference_scheduler_disagg_decision_total': 'count',
}

# Byte-unit conversions applied during parsing
_BYTE_CONVERSIONS = {
    'container_memory_usage_bytes': (1024**3, '_bytes', '_gb'),
    'container_memory_working_set_bytes': (1024**3, '_bytes', '_gb'),
    'container_network_receive_bytes_total': (1024**2, '_bytes_total', '_mb_total'),
    'container_network_transmit_bytes_total': (1024**2, '_bytes_total', '_mb_total'),
}


# NIXL metrics used for transfer characteristics regression
_NIXL_METRICS = {
    'vllm:nixl_xfer_time_seconds_sum',
    'vllm:nixl_xfer_time_seconds_count',
    'vllm:nixl_bytes_transferred_sum',
    'vllm:nixl_bytes_transferred_count',
}

MIN_REGRESSION_POINTS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(filepath):
    """Load a JSON file, returning {} if it doesn't exist."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)


def _save_json(filepath, data):
    """Save data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def percentile(sorted_values, p):
    """Calculate the p-th percentile from a sorted list using linear interpolation."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return sorted_values[0]
    k = (n - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, n - 1)
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def _compute_stats(values, unit=''):
    """Compute statistics dict from a list of numeric values."""
    sorted_vals = sorted(values)
    return {
        'mean': statistics.mean(values),
        'stddev': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'p25': percentile(sorted_vals, 25),
        'p50': percentile(sorted_vals, 50),
        'p75': percentile(sorted_vals, 75),
        'p90': percentile(sorted_vals, 90),
        'p95': percentile(sorted_vals, 95),
        'p99': percentile(sorted_vals, 99),
        'max': max(values),
        'count': len(values),
        'unit': unit,
    }


def _linear_regression(xs, ys):
    """Ordinary least-squares linear regression (no numpy dependency).

    Returns (slope, intercept, r_squared) or (None, None, 0.0) on failure.
    """
    n = len(xs)
    if n < MIN_REGRESSION_POINTS:
        return None, None, 0.0
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)
    denom = n * sum_x2 - sum_x ** 2
    if denom == 0:
        return None, None, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r_squared


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_prometheus_metrics(file_path):
    """Parse Prometheus metrics from a file."""
    metrics = defaultdict(list)
    timestamp = None
    pod_name = None
    namespace = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Extract metadata
            if line.startswith('# Timestamp:'):
                timestamp = line.split(':', 1)[1].strip()
            elif line.startswith('# Pod:'):
                pod_name = line.split(':', 1)[1].strip()
            elif line.startswith('# Namespace:'):
                namespace = line.split(':', 1)[1].strip()

            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue

            match = re.match(
                r'([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?) ([\d.eE+-]+)(?:\s+\d+)?', line)
            if match:
                base_name = match.group(1).split('{')[0]
                value = float(match.group(2))

                # Apply byte-unit conversions
                if base_name in _BYTE_CONVERSIONS:
                    divisor, old_suffix, new_suffix = _BYTE_CONVERSIONS[base_name]
                    value = value / divisor
                    base_name = base_name.replace(old_suffix, new_suffix)

                metrics[base_name].append(value)

    return timestamp, pod_name, namespace, dict(metrics)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics():
    """Aggregate metrics from all collected files."""
    pod_metrics = defaultdict(lambda: defaultdict(list))
    pod_metadata = {}

    all_files = glob.glob(os.path.join(raw_dir, '*_metrics.log'))

    if not all_files:
        print("Warning: No raw files found to process")
        print(f"Checked directory: {raw_dir}")
        results = {
            '_info': {
                'status': 'no_data',
                'message': 'No metrics collected - no raw files found',
                'raw_dir': raw_dir,
                'possible_reasons': [
                    'Metrics collection could not find any pods',
                    'kubectl/oc command may not have access to the cluster',
                    'Label selector may not match any pods',
                    'Namespace may be incorrect'
                ]
            }
        }
        _save_json(os.path.join(processed_dir, 'metrics_summary.json'), results)
        print(f"Empty metrics summary saved")
        return results

    print(f"Processing {len(all_files)} files...")

    for file_path in all_files:
        timestamp, pod_name, namespace, metrics = parse_prometheus_metrics(
            file_path)

        if pod_name:
            if pod_name not in pod_metadata:
                pod_metadata[pod_name] = {'namespace': namespace, 'files': []}
            pod_metadata[pod_name]['files'].append(os.path.basename(file_path))

            for metric_name, values in metrics.items():
                pod_metrics[pod_name][metric_name].extend(values)

    # Compute ratio metrics per-pod before aggregation
    for pod_name, metrics in pod_metrics.items():
        for ratio_name, num_metric, den_metric in RATIO_METRICS:
            if num_metric in metrics and den_metric in metrics:
                num_vals = metrics[num_metric]
                den_vals = metrics[den_metric]
                ratio_vals = [
                    (num_vals[i] / den_vals[i] * 100) if den_vals[i] > 0 else 0.0
                    for i in range(min(len(num_vals), len(den_vals)))
                ]
                if ratio_vals:
                    metrics[ratio_name] = ratio_vals

    # Per-pod statistics
    results = {}
    for pod_name, metrics in pod_metrics.items():
        results[pod_name] = {
            'metadata': pod_metadata.get(pod_name, {}),
            'metrics': {
                name: _compute_stats(values, METRIC_UNITS.get(name, ''))
                for name, values in metrics.items()
                if values
            }
        }

    # Cluster-wide aggregated statistics
    aggregated_values = defaultdict(list)
    for metrics in pod_metrics.values():
        for name, values in metrics.items():
            if name in AGGREGATE_METRICS and values:
                aggregated_values[name].extend(values)

    if aggregated_values:
        results['_aggregated'] = {
            'metrics': {
                name: _compute_stats(values, METRIC_UNITS.get(name, ''))
                for name, values in aggregated_values.items()
            }
        }
        print(f"Aggregated {len(aggregated_values)} metrics across all pods")

    output_file = os.path.join(processed_dir, 'metrics_summary.json')
    _save_json(output_file, results)
    print(f"Metrics summary saved to: {output_file}")
    print(f"Processed metrics from {len(results)} pods")
    return results


def aggregate_pod_startup_stats():
    """Compute aggregate statistics for pod startup times."""
    startup_file = os.path.join(processed_dir, 'pod_startup_times.json')
    data = _load_json(startup_file)

    values = [
        p['startup_seconds'] for p in data.get('pods', [])
        if isinstance(p.get('startup_seconds'), (int, float))
    ]
    if not values:
        return

    data['aggregate'] = _compute_stats(values, 's')
    _save_json(startup_file, data)
    print(f"Pod startup stats: {len(values)} pods, "
          f"mean={data['aggregate']['mean']:.1f}s")


def aggregate_replica_stats():
    """Compute aggregate statistics from replica status time series."""
    ts_data = _load_json(
        os.path.join(processed_dir, 'replica_status_timeseries.json'))
    snapshots = ts_data.get('snapshots', [])
    if not snapshots:
        return

    ready_counts = [
        sum(c.get('ready_replicas', 0) for c in snap.get('controllers', []))
        for snap in snapshots
    ]

    status_file = os.path.join(processed_dir, 'replica_status.json')
    status_data = _load_json(status_file)
    if status_data:
        status_data['aggregate_ready_replicas'] = _compute_stats(
            ready_counts, 'count')
        _save_json(status_file, status_data)

    print(f"Replica stats: {len(snapshots)} snapshots, "
          f"ready replicas min={min(ready_counts)} max={max(ready_counts)} "
          f"mean={statistics.mean(ready_counts):.1f}")


def _parse_nixl_time_series():
    """Re-parse raw files to extract time-ordered NIXL counter values per pod.

    Returns {pod_name: {metric_name: [(timestamp_str, value), ...]}} sorted by time.
    """
    pod_ts = defaultdict(lambda: defaultdict(list))

    for file_path in glob.glob(os.path.join(raw_dir, '*_metrics.log')):
        timestamp = None
        pod_name = None
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# Timestamp:'):
                    timestamp = line.split(':', 1)[1].strip()
                elif line.startswith('# Pod:'):
                    pod_name = line.split(':', 1)[1].strip()
                if line.startswith('#') or not line:
                    continue
                match = re.match(
                    r'([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?) ([\d.eE+-]+)',
                    line)
                if match and timestamp and pod_name:
                    base_name = match.group(1).split('{')[0]
                    if base_name in _NIXL_METRICS:
                        pod_ts[pod_name][base_name].append(
                            (timestamp, float(match.group(2))))

    for pod in pod_ts.values():
        for metric in pod.values():
            metric.sort(key=lambda x: x[0])

    return dict(pod_ts)


def _compute_transfer_characteristics(pod_ts):
    """Compute per-interval deltas and run linear regression.

    For each interval where transfers occurred, derives:
      avg_bytes_per_transfer and avg_time_per_transfer
    then fits: time = base_latency + bytes / bandwidth

    Returns {pod_name: {bandwidth_gbps, base_latency_ms, r_squared, num_data_points}}
    """
    results = {}

    for pod_name, metrics in pod_ts.items():
        bytes_sum = metrics.get('vllm:nixl_bytes_transferred_sum', [])
        time_sum = metrics.get('vllm:nixl_xfer_time_seconds_sum', [])
        count_series = metrics.get('vllm:nixl_bytes_transferred_count', [])

        if not bytes_sum or not time_sum or not count_series:
            continue

        ts_bytes = {ts: v for ts, v in bytes_sum}
        ts_time = {ts: v for ts, v in time_sum}
        ts_count = {ts: v for ts, v in count_series}

        common_ts = sorted(set(ts_bytes) & set(ts_time) & set(ts_count))
        if len(common_ts) < 2:
            continue

        xs_bytes = []
        ys_time = []
        for i in range(1, len(common_ts)):
            prev, cur = common_ts[i - 1], common_ts[i]
            delta_bytes = ts_bytes[cur] - ts_bytes[prev]
            delta_time = ts_time[cur] - ts_time[prev]
            delta_count = ts_count[cur] - ts_count[prev]

            if delta_count > 0 and delta_bytes > 0 and delta_time > 0:
                xs_bytes.append(delta_bytes / delta_count)
                ys_time.append(delta_time / delta_count)

        slope, intercept, r_squared = _linear_regression(xs_bytes, ys_time)
        if slope is None or slope <= 0:
            continue

        results[pod_name] = {
            'bandwidth_gbps': round(1.0 / slope / 1e9, 4),
            'base_latency_ms': round(intercept * 1000, 4),
            'r_squared': round(r_squared, 4),
            'num_data_points': len(xs_bytes),
        }

    return results


def compute_nixl_transfer_characteristics():
    """Top-level function to derive NIXL transfer bandwidth and base latency."""
    pod_ts = _parse_nixl_time_series()
    if not pod_ts:
        return

    per_pod = _compute_transfer_characteristics(pod_ts)
    if not per_pod:
        print("NIXL transfer characteristics: insufficient data for regression")
        return

    # Aggregate across all pods
    all_bytes = []
    all_time = []
    for pod_name, metrics in pod_ts.items():
        bytes_sum = metrics.get('vllm:nixl_bytes_transferred_sum', [])
        time_sum = metrics.get('vllm:nixl_xfer_time_seconds_sum', [])
        count_series = metrics.get('vllm:nixl_bytes_transferred_count', [])
        if not bytes_sum or not time_sum or not count_series:
            continue
        ts_bytes = {ts: v for ts, v in bytes_sum}
        ts_time = {ts: v for ts, v in time_sum}
        ts_count = {ts: v for ts, v in count_series}
        common_ts = sorted(set(ts_bytes) & set(ts_time) & set(ts_count))
        for i in range(1, len(common_ts)):
            prev, cur = common_ts[i - 1], common_ts[i]
            db = ts_bytes[cur] - ts_bytes[prev]
            dt = ts_time[cur] - ts_time[prev]
            dc = ts_count[cur] - ts_count[prev]
            if dc > 0 and db > 0 and dt > 0:
                all_bytes.append(db / dc)
                all_time.append(dt / dc)

    slope, intercept, r_squared = _linear_regression(all_bytes, all_time)
    aggregated = None
    if slope and slope > 0:
        aggregated = {
            'bandwidth_gbps': round(1.0 / slope / 1e9, 4),
            'base_latency_ms': round(intercept * 1000, 4),
            'r_squared': round(r_squared, 4),
            'num_data_points': len(all_bytes),
        }

    # Merge into metrics_summary.json
    summary_file = os.path.join(processed_dir, 'metrics_summary.json')
    summary = _load_json(summary_file)

    for pod_name, chars in per_pod.items():
        if pod_name in summary:
            summary[pod_name].setdefault('metrics', {})[
                'nixl_transfer_characteristics'] = chars

    if aggregated:
        summary.setdefault('_aggregated', {}).setdefault('metrics', {})[
            'nixl_transfer_characteristics'] = aggregated

    _save_json(summary_file, summary)

    for pod_name, chars in per_pod.items():
        print(f"NIXL transfer [{pod_name}]: "
              f"bandwidth={chars['bandwidth_gbps']} GB/s, "
              f"base_latency={chars['base_latency_ms']} ms, "
              f"R²={chars['r_squared']} ({chars['num_data_points']} points)")
    if aggregated:
        print(f"NIXL transfer [aggregated]: "
              f"bandwidth={aggregated['bandwidth_gbps']} GB/s, "
              f"base_latency={aggregated['base_latency_ms']} ms, "
              f"R²={aggregated['r_squared']} ({aggregated['num_data_points']} points)")


if __name__ == '__main__':
    aggregate_metrics()
    aggregate_pod_startup_stats()
    aggregate_replica_stats()
    compute_nixl_transfer_characteristics()
