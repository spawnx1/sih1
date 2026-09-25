"""
generator/ -- standalone SYNTHETIC cyber-fraud case generator for SIH26184.

Everything this package produces is clearly synthetic (ACC_SYN_*, CELL_*,
CASHOUT_SYN_*). It never touches real banking data, real people, real accounts
or real credentials. Its only job is to manufacture leakage-safe demo/test
cases that our existing ML + Graph + GNN/TGN pipeline can reconstruct and
forecast.

Entry point:  python -m generator.case --scenario cashout --seed 42

Import the API with:  from generator.case import generate_case, SCENARIOS
(kept out of this __init__ so `python -m generator.case` doesn't double-import.)
"""
