from json import load
from collections.abc import MutableMapping
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key.replace("procesaur/", "") + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, np.array(v, dtype=float)))
    return dict(items)


with open("data/probabilities.json", "r") as jf:
    probs = flatten_dict(load(jf))

cols_sr = [x for x in probs if ".sr" in x]
cols_bad = [x for x in probs if ".bad" in x]
cols_google = [x for x in probs if ".google" in x]

cols = [cols_sr, cols_bad, cols_google, probs]

for col in cols:
    df = pd.DataFrame(0, columns=col, index=col)
    for x in col:
        for y in col:
            if x == y:
                df[x][y] = 1
            else:
                if df[y][x] > 0:
                    df[x][y] = df[y][x]
                else:
                    p, _ = pearsonr(probs[x], probs[y])
                    df.loc[x, y] = p
    print(df.to_string())
