from json import load
from collections.abc import MutableMapping
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from data_preprocessing import map_inner, map_outer


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = map_outer(parent_key) + sep + map_inner(k) if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, np.array(v, dtype=float)))
    return dict(items)


def fill_and_print(cols):
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
                        df.loc[x, y] = round(p, 3)
        print(df.to_csv(sep="\t"))


with open("data/probabilities.json", "r") as jf:
    probs = flatten_dict(load(jf))

cols_t1 = sorted([x for x in probs if ".t1" in x])
cols_t2 = sorted([x for x in probs if ".t2" in x])
cols_t3 = sorted([x for x in probs if ".t3" in x])
cols_m1 = sorted([x for x in probs if "m1." in x])
cols_m2 = sorted([x for x in probs if "m2." in x])
cols_m3 = sorted([x for x in probs if "m3." in x])

fill_and_print([cols_t1, cols_t2, cols_t3])
fill_and_print([cols_m1, cols_m2, cols_m3])
