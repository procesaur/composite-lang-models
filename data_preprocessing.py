from json import load, dump


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


def data_preprocessing(file):
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

    dataset = {
        "models": {i: m for i, m in enumerate(models)},
        "sets": {i: s for i, s in enumerate(sets)},
        "data": data
    }

    with open(file.replace(".json", "_processed.json"), "w") as jf2:
        dump(dataset, jf2, ensure_ascii=False)


file_path = "data/probabilities.json"
data_preprocessing(file_path)

file_path = "data/prob_vectors.json"
data_preprocessing(file_path)
