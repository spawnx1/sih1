"""
tests/test_leakage.py -- the mechanical guarantees behind R1/R2/R3.

These are the tests that make "no leakage" a property of the code rather than a
promise. They assert:

  1. AS-OF (R1): recomputing every feature against a corpus PHYSICALLY truncated
     at as_of gives bit-identical values. If a feature secretly peeked at a
     future row, truncating that row would change its value and this fails.
  2. SIGNATURE (R1): every feature function is f(account_id, as_of, ctx).
  3. NO SINGLE-FEATURE LEAK: no feature correlates with the label above
     |r| = 0.9 on the panel.
  4. LOADER (R2/R3): no `label_` or internal column survives the feature loader.
  5. IMPORT WALL: nothing under ml/ or graphx/ imports gen.params (the ML side
     must not read the generator's answers).

Run:  pytest tests/ -v
"""
from __future__ import annotations

import inspect
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

import config
from ml.features import (FeatureContext, FEATURE_FUNCS, drop_leaky_columns,
                         feature_names)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def ctx():
    c = FeatureContext(config.DATA_DIR)
    import joblib
    scorer_path = os.path.join(config.MODEL_DIR, "mule_scorer.joblib")
    if os.path.exists(scorer_path):
        c.mule_scorer = joblib.load(scorer_path)
    else:
        from graphx.features import MuleScorer
        c.mule_scorer = MuleScorer().fit(c)
    return c


def _sample_accounts(ctx, n=4):
    """Accounts with enough history to exercise the feature functions."""
    out = []
    for acc in ctx.accounts["account_id"]:
        pos = ctx.src_pos.get(acc)
        if pos is not None and len(pos) >= 6:
            out.append(acc)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------
# 1. AS-OF: physical-truncation bit-identity (R1)
# --------------------------------------------------------------------------
def test_as_of_truncation_bit_identical(ctx):
    accts = _sample_accounts(ctx, 4)
    assert accts, "no sampled accounts"
    # choose an as_of in the middle of the corpus
    as_of = pd.Timestamp(np.quantile(ctx.ts.astype("int64"), 0.6)).floor("s").to_pydatetime()

    tmp = tempfile.mkdtemp(prefix="asof_")
    try:
        # physically truncate the corpus to ts < as_of and rebuild a context
        tx = ctx.tx[ctx.tx["ts"] < pd.Timestamp(as_of)].copy()
        tx.to_parquet(os.path.join(tmp, "transactions.parquet"), index=False)
        ctx.atms.to_parquet(os.path.join(tmp, "atms.parquet"), index=False)
        ctx.accounts.to_parquet(os.path.join(tmp, "accounts.parquet"), index=False)
        ctx_tr = FeatureContext(tmp)
        ctx_tr.mule_scorer = ctx.mule_scorer

        ctx.bind(None, None)
        ctx_tr.bind(None, None)
        for acc in accts:
            for name, fn in FEATURE_FUNCS.items():
                v_full = fn(acc, as_of, ctx)
                v_tr = fn(acc, as_of, ctx_tr)
                assert np.isclose(v_full, v_tr, rtol=0, atol=1e-9), (
                    f"feature {name} changed under truncation for {acc}: "
                    f"{v_full} != {v_tr} -- it peeked past as_of")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# 2. SIGNATURE: every feature is f(account_id, as_of, ctx) (R1)
# --------------------------------------------------------------------------
def test_feature_signatures():
    for name, fn in FEATURE_FUNCS.items():
        params = list(inspect.signature(fn).parameters)
        assert params == ["a", "as_of", "ctx"] or params[:3] == ["a", "as_of", "ctx"], (
            f"{name} has signature {params}, expected (a, as_of, ctx)")


# --------------------------------------------------------------------------
# 3. no single feature correlates with the label above |r| = 0.9
# --------------------------------------------------------------------------
def test_no_feature_correlates_above_0p9():
    panel_path = os.path.join(config.MODEL_DIR, "panel_train.parquet")
    if not os.path.exists(panel_path):
        pytest.skip("panel_train.parquet not found; run ml.train first")
    df = pd.read_parquet(panel_path)
    y = df["y"].to_numpy(dtype=float)
    offenders = {}
    for f in feature_names() + ["horizon_min"]:
        if f not in df.columns:
            continue
        x = df[f].to_numpy(dtype=float)
        if np.std(x) < 1e-12:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if abs(r) > 0.9:
            offenders[f] = round(float(r), 3)
    assert not offenders, f"features leak (|r|>0.9 with label): {offenders}"


# --------------------------------------------------------------------------
# 4. LOADER: label_ / internal columns never survive (R2/R3)
# --------------------------------------------------------------------------
def test_loader_drops_label_and_internal():
    df = pd.DataFrame({
        "taint_amt": [1.0], "label_is_mule": [1], "label_case_id": ["x"],
        "archetype": ["MULE_RING"], "home_lat": [24.0], "home_lon": [86.0],
        "home_cell": ["abc"], "hop_from_victim": [2.0],
    })
    out = drop_leaky_columns(df)
    for bad in ("label_is_mule", "label_case_id", "archetype", "home_lat",
                "home_lon", "home_cell"):
        assert bad not in out.columns, f"{bad} survived the feature loader"
    for good in ("taint_amt", "hop_from_victim"):
        assert good in out.columns


# --------------------------------------------------------------------------
# 5. IMPORT WALL: ml/ and graphx/ must not import gen.params
# --------------------------------------------------------------------------
def test_ml_side_does_not_import_generator_params():
    import re
    pattern = re.compile(r"import\s+gen\.params|from\s+gen\.params|from\s+gen\s+import\s+params")
    offenders = []
    for pkg in ("ml", "graphx"):
        d = os.path.join(HERE, pkg)
        for fn in os.listdir(d):
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                if pattern.search(f.read()):
                    offenders.append(f"{pkg}/{fn}")
    assert not offenders, f"ML side imports generator params (cheating): {offenders}"
