from __future__ import annotations

import numpy as np
import pandas as pd

import config
from graphx.candidates import generate_candidates
from ml.features import FeatureContext
from ml.train import build_m3_events


def evaluate(ctx, events, name):
    rows = []

    for i, ev in enumerate(events):
        true_atm = str(ev["terminal"])

        if true_atm == "None" or true_atm not in ctx.atm.index:
            continue

        candidates = [
            str(x)
            for x in generate_candidates(
                ev["account"],
                ev["as_of"],
                ctx
            )
        ]

        rank_map = {atm: j + 1 for j, atm in enumerate(candidates)}
        rank = rank_map.get(true_atm)

        rows.append({
            "event": i,
            "account": str(ev["account"]),
            "as_of": ev["as_of"],
            "true_atm": true_atm,
            "has_hist": bool(ev["has_hist"]),
            "candidate_count": len(candidates),
            "rank": rank if rank is not None else 999999,
        })

    df = pd.DataFrame(rows)

    print()
    print(name)
    print("-" * 70)
    print(f"Events evaluated : {len(df):,}")

    if df.empty:
        print("NO VALID EVENTS")
        return df

    for k in [1, 3, 5, 10, 50]:
        hit = (df["rank"] <= k).mean()
        print(f"Recall@{k:<2}        : {hit:.4f}")

    print(f"Median candidates: {df['candidate_count'].median():.1f}")

    valid_rank = df.loc[df["rank"] < 999999, "rank"]

    if len(valid_rank):
        print(f"Median rank      : {valid_rank.median():.1f}")
    else:
        print("Median rank      : N/A")

    for label, mask in [
        ("WITH prior ATM history", df["has_hist"]),
        ("WITHOUT prior ATM history", ~df["has_hist"]),
    ]:
        sub = df[mask]

        if len(sub):
            print()
            print(f"{label} (n={len(sub):,})")
            print(f"Recall@50        : {(sub['rank'] <= 50).mean():.4f}")
            print(f"Recall@10        : {(sub['rank'] <= 10).mean():.4f}")
            print(f"Recall@5         : {(sub['rank'] <= 5).mean():.4f}")

    return df


def main():
    print("=" * 70)
    print("SPRINT 9A - CASH-OUT ATM CANDIDATE CEILING")
    print("=" * 70)

    ctx = FeatureContext(config.DATA_DIR)

    print()
    print("BUILDING CASH-OUT EVENTS...")

    events = build_m3_events(
        ctx,
        seed=config.SEED,
        n_legit=3000
    )

    tainted = [
        e for e in events
        if e["is_tainted"]
    ]

    print(f"Total events      : {len(events):,}")
    print(f"Tainted cash-outs : {len(tainted):,}")

    if not tainted:
        raise RuntimeError("No tainted cash-out events found.")

    times = pd.Series([
        e["as_of"]
        for e in tainted
    ])

    cut = times.quantile(0.70)

    train_events = [
        e for e in tainted
        if e["as_of"] < cut
    ]

    test_events = [
        e for e in tainted
        if e["as_of"] >= cut
    ]

    print()
    print("TEMPORAL SPLIT")
    print(f"Cutoff            : {cut}")
    print(f"Train events      : {len(train_events):,}")
    print(f"Test events       : {len(test_events):,}")

    evaluate(
        ctx,
        train_events,
        "TRAIN CANDIDATE RECALL"
    )

    test_df = evaluate(
        ctx,
        test_events,
        "TEST CANDIDATE RECALL"
    )

    print()
    print("=" * 70)
    print("SPRINT 9A RESULT")
    print("=" * 70)

    if len(test_df):
        r50 = (test_df["rank"] <= 50).mean()
        r10 = (test_df["rank"] <= 10).mean()
        r5 = (test_df["rank"] <= 5).mean()
        r1 = (test_df["rank"] <= 1).mean()

        print(f"Candidate Recall@50 : {r50:.4f}")
        print(f"Candidate Recall@10 : {r10:.4f}")
        print(f"Candidate Recall@5  : {r5:.4f}")
        print(f"Candidate Recall@1  : {r1:.4f}")

        if r50 >= 0.80:
            print()
            print("CANDIDATE CEILING: READY FOR RANKING")
        else:
            print()
            print("CANDIDATE CEILING: NEEDS IMPROVEMENT")

    print()
    print("SPRINT 9A COMPLETE")


if __name__ == "__main__":
    main()
