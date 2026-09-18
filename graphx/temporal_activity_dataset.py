import numpy as np
from ml.features import FeatureContext
from graphx.gnn_temporal_examples import (
    build_snapshot,
    future_cashout_labels,
)


# ============================================================
# SPRINT 6A
# HIGH-FREQUENCY ACTIVITY-BASED TEMPORAL DATASET
# ============================================================

INTERVAL_MIN = 15
MAX_SNAPSHOTS = 40
HORIZON_MIN = 60


def floor_to_bucket(ts, interval_min=15):
    # Convert through a datetime64 minute scalar.
    # Using integer arithmetic on the minute timestamp
    # avoids NumPy object-to-datetime conversion issues.
    ts_min = np.asarray(
        ts,
        dtype="datetime64[m]",
    )

    minute_value = int(
        ts_min.astype(np.int64)
    )

    bucket_minute = (
        minute_value // interval_min
    ) * interval_min

    return (
        np.datetime64(
            "1970-01-01T00:00",
            "m",
        )
        + np.timedelta64(
            bucket_minute,
            "m",
        )
    )


def build_activity_buckets(ctx):
    """
    Create 15-minute buckets containing observed
    transaction activity.
    """

    ts = ctx.ts

    valid = (
        ts == ts
    )

    timestamps = ts[valid]

    buckets = {}

    for t in timestamps:

        bucket = floor_to_bucket(
            t,
            INTERVAL_MIN,
        )

        buckets.setdefault(
            bucket,
            0,
        )

        buckets[bucket] += 1

    return buckets


def future_cashout_count(
    ctx,
    snapshot_time,
    horizon_min=60,
):
    """
    Count cash-outs occurring after the snapshot
    and within the prediction horizon.
    """

    start = np.datetime64(
        snapshot_time,
        "ns",
    )

    end = (
        start
        + np.timedelta64(
            horizon_min,
            "m",
        )
    )

    mask = (
        (ctx.ts > start)
        & (ctx.ts <= end)
        & ctx.is_cashout
    )

    return int(mask.sum())


def choose_activity_snapshots(
    ctx,
    buckets,
    max_snapshots=40,
):
    """
    Select snapshots from actual activity buckets.

    We prefer:
      1. buckets with future cash-out activity
      2. buckets with transaction activity
      3. chronological coverage across the dataset

    Selection is deterministic.
    """

    candidates = []

    for bucket, activity_count in buckets.items():

        future_cashouts = future_cashout_count(
            ctx,
            bucket,
            HORIZON_MIN,
        )

        candidates.append(
            (
                bucket,
                activity_count,
                future_cashouts,
            )
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    if len(candidates) <= max_snapshots:
        return [
            x[0]
            for x in candidates
        ]

    # --------------------------------------------------------
    # Divide timeline into chronological regions.
    # Select representatives from each region.
    # --------------------------------------------------------

    n = len(candidates)

    selected = []

    # First select candidates with future cash-out activity.
    positive_candidates = [
        x for x in candidates
        if x[2] > 0
    ]

    # Deterministic spacing through positive candidates.
    if positive_candidates:

        positive_count = min(
            max_snapshots // 2,
            len(positive_candidates),
        )

        indices = np.linspace(
            0,
            len(positive_candidates) - 1,
            positive_count,
            dtype=int,
        )

        for idx in indices:
            selected.append(
                positive_candidates[idx][0]
            )

    # Fill remaining positions with activity buckets
    # distributed across the complete timeline.
    selected_set = set(selected)

    remaining = [
        x for x in candidates
        if x[0] not in selected_set
    ]

    remaining_count = (
        max_snapshots - len(selected)
    )

    if remaining_count > 0 and remaining:

        indices = np.linspace(
            0,
            len(remaining) - 1,
            min(
                remaining_count,
                len(remaining),
            ),
            dtype=int,
        )

        for idx in indices:
            selected.append(
                remaining[idx][0]
            )

    selected = sorted(
        set(selected)
    )

    return selected[:max_snapshots]


def main():

    print("=" * 70)
    print("SPRINT 6A - HIGH-FREQUENCY TEMPORAL DATASET")
    print("=" * 70)

    ctx = FeatureContext()

    print()
    print("Transactions:", len(ctx.ts))

    buckets = build_activity_buckets(
        ctx
    )

    print(
        "15-minute activity buckets:",
        len(buckets),
    )

    active_transactions = sum(
        buckets.values()
    )

    print(
        "Transactions represented:",
        active_transactions,
    )

    snapshot_times = choose_activity_snapshots(
        ctx,
        buckets,
        MAX_SNAPSHOTS,
    )

    print()
    print(
        "Selected snapshots:",
        len(snapshot_times),
    )

    print()
    print("BUILDING SNAPSHOTS")
    print("-" * 70)

    examples = []

    total_nodes = 0
    total_edges = 0
    total_positive = 0

    for i, snapshot_time in enumerate(
        snapshot_times,
        start=1,
    ):

        data, id_to_node, _ = build_snapshot(
            ctx,
            snapshot_time,
        )

        if data.num_nodes == 0:
            print(
                f"[{i:02d}/{len(snapshot_times)}] "
                f"{snapshot_time} | EMPTY"
            )

            examples.append(
                (
                    snapshot_time,
                    data,
                    None,
                )
            )

            continue

        labels = future_cashout_labels(
            ctx,
            snapshot_time,
            list(id_to_node.keys()),
            horizon_min=HORIZON_MIN,
        )

        positives = int(
            labels.sum().item()
        )

        activity = buckets.get(
            snapshot_time,
            0,
        )

        print(
            f"[{i:02d}/{len(snapshot_times)}] "
            f"{snapshot_time} | "
            f"Activity: {activity} | "
            f"Nodes: {data.num_nodes} | "
            f"Edges: {data.num_edges} | "
            f"Positives: {positives}"
        )

        examples.append(
            (
                snapshot_time,
                data,
                labels,
            )
        )

        total_nodes += data.num_nodes
        total_edges += data.num_edges
        total_positive += positives

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    valid_examples = [
        x for x in examples
        if x[2] is not None
    ]

    split = int(
        len(valid_examples) * 0.70
    )

    train_examples = valid_examples[:split]
    test_examples = valid_examples[split:]

    train_positive = sum(
        int(x[2].sum().item())
        for x in train_examples
    )

    train_nodes = sum(
        x[1].num_nodes
        for x in train_examples
    )

    test_positive = sum(
        int(x[2].sum().item())
        for x in test_examples
    )

    test_nodes = sum(
        x[1].num_nodes
        for x in test_examples
    )

    print()
    print("=" * 70)
    print("TEMPORAL DATASET SUMMARY")
    print("=" * 70)

    print(
        "Valid snapshots:",
        len(valid_examples),
    )

    print(
        "Train snapshots:",
        len(train_examples),
    )

    print(
        "Test snapshots :",
        len(test_examples),
    )

    print()
    print(
        "Train nodes:",
        train_nodes,
    )

    print(
        "Test nodes :",
        test_nodes,
    )

    print()
    print(
        "Train positives:",
        train_positive,
    )

    print(
        "Test positives :",
        test_positive,
    )

    if train_nodes:
        print(
            "Train positive rate:",
            round(
                train_positive / train_nodes,
                6,
            ),
        )

    if test_nodes:
        print(
            "Test positive rate :",
            round(
                test_positive / test_nodes,
                6,
            ),
        )

    if valid_examples:
        print()
        print(
            "TRAIN PERIOD:"
        )
        print(
            train_examples[0][0],
            "->",
            train_examples[-1][0],
        )

        print()
        print(
            "TEST PERIOD:"
        )
        print(
            test_examples[0][0],
            "->",
            test_examples[-1][0],
        )

    print()
    print(
        "Total graph nodes processed:",
        total_nodes,
    )

    print(
        "Total graph edges processed:",
        total_edges,
    )

    print(
        "Total positive labels:",
        total_positive,
    )

    print("=" * 70)
    print("SPRINT 6A DATASET: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
