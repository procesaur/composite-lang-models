from json import load, dump
import numpy as np
import scipy.stats as stats
from scipy.fft import fft
from tqdm import tqdm


def map_outer(key):
    if key == "procesaur/gpt2-srlat":
        return "m1"
    if key == "procesaur/gpt2-srlat-sem":
        return "m2"
    if key == "procesaur/gpt2-srlat-synt":
        return "m3"


def map_inner(key):
    if key == "sr":
        return "t1"
    if key == "bad":
        return "t2"
    if key == "google":
        return "t3"


def pairwise(lst):
    it = iter(lst)
    a = next(it, None)
    for b in it:
        yield a, b
        a = b


def pairwise_sub(lst):
    out = []
    for x in lst:
        out.append([round(b-a, 14) for a, b in pairwise([0, *x])])
    return out


def features_extraction(arr):
    features = []
    values = []
    for a in arr:
        if len(a) < 1:
            a = [0]
        a = np.array(a)
        # TIME DOMAIN
        features.extend(['MIN', 'MAX', 'MEAN', 'RMS', 'VAR', 'STD', 'POWER', 'PEAK', 'P2P', 'CREST FACTOR', 'SKEW',
                         'KURTOSIS', 'FORM_f', 'Pulse'])
        values.extend([np.min(a),
                       np.max(a),
                       np.mean(a),
                       np.sqrt(np.mean(a ** 2)),
                       np.var(a),
                       np.std(a),
                       np.mean(a ** 2),
                       np.max(np.abs(a)),
                       np.ptp(a),
                       np.max(np.abs(a)) / np.sqrt(np.mean(a ** 2)),
                       stats.skew(a),
                       stats.kurtosis(a),
                       np.sqrt(np.mean(a ** 2)) / np.mean(a),
                       np.max(np.abs(a)) / np.mean(a)])

        # FREQ DOMAIN
        fourier = fft(a)
        spectrum = np.abs(fourier ** 2) / len(a)
        features.extend(['MAX_f', 'SUM_f', 'MEAN_f', 'VAR_f', 'PEAK_f', 'SKEW_f', 'KURTOSIS_f'])
        values.extend([np.max(spectrum),
                       np.sum(spectrum),
                       np.mean(spectrum),
                       np.var(spectrum),
                       np.max(np.abs(spectrum)),
                       stats.skew(a),
                       stats.kurtosis(a)])

    for i, val in enumerate(values):
        if np.isnan(val):
            values[i] = float(0)
        else:
            values[i] = float(round(val, 14))

    return features, values


def data_preprocessing(file, added_features=False, transform_vectors=False):
    with open(file, "r") as jf:
        loaded = load(jf)
    filename = file.replace(".json", "_processed.json")
    filename_add = file.replace(".json", "_processed_addon.json")
    models = sorted([map_outer(x) for x in loaded.keys()])
    loaded = {map_outer(key): value for (key, value) in loaded.items()}
    sets = []
    [sets.extend(loaded[model].keys()) for model in models]
    sets = sorted([map_inner(x) for x in list(set(sets))])
    loaded = {model: {map_inner(x): values for x, values in sets.items()} for model, sets in loaded.items()}

    values = [list(map(list, zip(*[loaded[x][y] for x in models]))) for y in sets]
    data = []

    for cls, lst in enumerate(values):
        data.extend([(cls, x) for x in values[cls]])

    if transform_vectors:
        filename = file.replace(".json", "_processed_2.json")
        filename_add = file.replace(".json", "_processed_2_addon.json")
        for i, line in enumerate(data):
            data[i] = (data[i][0], pairwise_sub(data[i][1]))

    dataset = {
        "models": {i: m for i, m in enumerate(models)},
        "sets": {i: s for i, s in enumerate(sets)},
        "data": data
    }

    with open(filename, "w") as jf2:
        dump(dataset, jf2, ensure_ascii=False)

    if added_features:
        data2 = []
        features = []

        with open(file_path1.replace(".json", "_processed.json"), "r") as jf:
            processed_data = load(jf)["data"]

        for i, sample in tqdm(enumerate(data), total=len(data)):
            features, values = features_extraction(sample[1])
            values.extend(processed_data[i][1])
            data2.append(values)

        dataset2 = {
            "features": {i: f for i, f in enumerate(features)},
            "data": data2
        }

        with open(filename_add, "w") as jf2:
            dump(dataset2, jf2, ensure_ascii=False)


file_path1 = "data/probabilities.json"
file_path2 = "data/prob_vectors.json"

data_preprocessing(file_path1)
data_preprocessing(file_path2, True)
data_preprocessing(file_path2, True, True)
