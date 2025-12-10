"""Compare multiple submission files and show prediction statistics."""
from pathlib import Path
import pandas as pd
import numpy as np

script_dir = Path(__file__).resolve().parent

# Load submissions
subs = {}
for f in ["submission.csv", "submission_ensemble_topk.csv", "submission_fold1params_fulltrain.csv"]:
    p = script_dir / f
    if p.exists():
        df = pd.read_csv(p, header=None, names=["id", "prob"])
        subs[f] = df
        print(f"{f}:")
        print(f"  Rows: {len(df)}")
        print(f"  Prob min/max/mean: {df['prob'].min():.6f} / {df['prob'].max():.6f} / {df['prob'].mean():.6f}")
        print(f"  Prob std: {df['prob'].std():.6f}")
        print()

# Compare pairs
if len(subs) >= 2:
    files = list(subs.keys())
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            f1, f2 = files[i], files[j]
            df1, df2 = subs[f1], subs[f2]
            # assume same order of ids
            if len(df1) == len(df2):
                diff = (df1["prob"] - df2["prob"]).abs()
                print(f"Difference {f1} vs {f2}:")
                print(f"  Mean abs diff: {diff.mean():.6f}")
                print(f"  Max abs diff: {diff.max():.6f}")
                print(f"  Correlation: {np.corrcoef(df1['prob'], df2['prob'])[0, 1]:.6f}")
                print()
