from json import load, dump
import numpy as np
import scipy.stats as stats
from scipy.fft import fft


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
        try:
            values[i] = float(val)
        except:
            print(val)

    return features, values


def data_preprocessing(file, added_features=False):
    with open(file, "r") as jf:
        loaded = load(jf)
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

    if added_features:
        with open(file_path1.replace(".json", "_processed.json"), "r") as jf:
            processed_data = load(jf)["data"]

        for i, sample in enumerate(data):
            features, values = features_extraction(sample[1])
            values.extend(processed_data[i][1])
            data[i][1].append(values)

    dataset = {
        "models": {i: m for i, m in enumerate(models)},
        "sets": {i: s for i, s in enumerate(sets)},
        "data": data
    }

    if added_features:
        dataset["features"] = {i: f for i, f in enumerate(features)}

    with open(file.replace(".json", "_processed.json"), "w") as jf2:
        dump(dataset, jf2, ensure_ascii=False)


file_path1 = "data/probabilities.json"
file_path2 = "data/prob_vectors.json"

# data_preprocessing(file_path1)
data_preprocessing(file_path2, True)
