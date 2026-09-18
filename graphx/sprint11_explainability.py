
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import FeatureContext


def explain_account(ctx, account_id, as_of):

    print()
    print("=" * 70)
    print("SPRINT 11 - ACCOUNT RISK EXPLANATION")
    print("=" * 70)

    print("Account:", account_id)
    print("As-of :", as_of)

    positions = ctx.out_positions(
        account_id,
        as_of,
        visible=True
    )

    in_positions = ctx.in_positions(
        account_id,
        as_of,
        visible=True
    )

    print()
    print("RELATIONSHIP SUMMARY")
    print("-" * 70)

    print(
        "Incoming transactions:",
        len(in_positions)
    )

    print(
        "Outgoing transactions:",
        len(positions)
    )

    outgoing_accounts = set()

    for pos in positions:

        dst = str(ctx.dst[pos])

        if dst != account_id:
            outgoing_accounts.add(dst)

    incoming_accounts = set()

    for pos in in_positions:

        src = str(ctx.src[pos])

        if src != account_id:
            incoming_accounts.add(src)

    print(
        "Unique incoming accounts:",
        len(incoming_accounts)
    )

    print(
        "Unique outgoing accounts:",
        len(outgoing_accounts)
    )

    withdrawals = []

    for pos in positions:

        if not ctx.is_cashout[pos]:
            continue

        atm = ctx.atm_id[pos]

        if atm is None:
            continue

        withdrawals.append({
            "atm": str(atm),
            "amount": float(ctx.amount[pos]),
            "ts": ctx.ts[pos]
        })

    print()
    print("HISTORICAL CASH-OUTS")
    print("-" * 70)

    if not withdrawals:

        print("No historical cash-outs found.")

    else:

        print(
            "Cash-out count:",
            len(withdrawals)
        )

        total = sum(
            x["amount"]
            for x in withdrawals
        )

        print(
            "Total cash-out amount:",
            round(total, 2)
        )

        for item in withdrawals[-5:]:

            print(
                f"ATM={item['atm']} "
                f"Amount={item['amount']:.2f} "
                f"Time={item['ts']}"
            )

    print()
    print("RISK SIGNALS")
    print("-" * 70)

    signals = []

    if len(in_positions) > 0:
        signals.append(
            f"Account has {len(in_positions)} historical incoming transactions."
        )

    if len(positions) > 0:
        signals.append(
            f"Account has {len(positions)} historical outgoing transactions."
        )

    if len(incoming_accounts) >= 3:
        signals.append(
            f"Connected to {len(incoming_accounts)} distinct incoming accounts."
        )

    if len(outgoing_accounts) >= 3:
        signals.append(
            f"Connected to {len(outgoing_accounts)} distinct outgoing accounts."
        )

    if withdrawals:
        signals.append(
            f"Historical cash-out activity observed at {len(set(x['atm'] for x in withdrawals))} ATM(s)."
        )

    if not signals:
        signals.append(
            "No simple graph risk signals detected."
        )

    for i, signal in enumerate(signals, 1):
        print(f"{i}. {signal}")

    print()
    print("INTERPRETATION")
    print("-" * 70)

    print(
        "The graph representation captures relationships between "
        "accounts and financial endpoints."
    )

    print(
        "These signals are evidence for investigation, not proof "
        "that an account is fraudulent."
    )


def main():

    ctx = FeatureContext()

    account_ids = sorted(
        set(str(x) for x in ctx.src)
        | set(str(x) for x in ctx.dst)
    )

    print("=" * 70)
    print("SPRINT 11 - GRAPH EXPLAINABILITY")
    print("=" * 70)

    print(
        "Accounts available:",
        len(account_ids)
    )

    # Select an account with the highest number of outgoing
    # historical transactions for a useful demonstration.
    best_account = None
    best_count = -1

    for account in account_ids:

        try:
            count = len(
                ctx.out_positions(
                    account,
                    ctx.ts[-1],
                    visible=True
                )
            )
        except Exception:
            continue

        if count > best_count:
            best_count = count
            best_account = account

    if best_account is None:
        print("No account available.")
        return

    print(
        "Demo account:",
        best_account
    )

    explain_account(
        ctx,
        best_account,
        ctx.ts[-1]
    )


if __name__ == "__main__":
    main()
