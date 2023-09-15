from models import Perce, Perceptron, CNNet
from json import load, dump


test_net_outputs = False
training_and_testing = True
cross_validation = 5


def save_results(dictionary):
    with open("data/results.json", "w") as rf:
        dump(dictionary, rf)


def get_nth_train_test(ordinal, cut, lst, classes):
    train_set = []
    test_set = []
    for c in classes:
        rows = [x for x in lst if x[0] == c]
        if max(classes) > len(classes)-1:
            class_map = {x: k for k, x in enumerate(classes)}
            rows = [[class_map[x[0]], x[1]] for x in rows]
        train_set.extend(rows[:ordinal*cut])
        train_set.extend(rows[(ordinal+1)*cut:])
        test_set.extend(rows[n*cut:(ordinal+1)*cut])
    return train_set, test_set


if test_net_outputs:
    model_path = "D:/aplikacije_mihailo/Parallel-language-models/nets/sem-ord.pt"
    inputs = [[0, 0.1, 0.1]]
    result = Perceptron(model_path).evaluate(inputs, True)
    print(result)

    model_path = "D:/aplikacije_mihailo/Parallel-language-models/nets/general_cnn.pt"
    inputs = [[[0.1, 0.2, 0.1], [0.1], [0.9]]]
    result = CNNet(model_path).evaluate(inputs, True)
    print(result)


if training_and_testing:
    results = {
        "m0": {},
        "m1": {},
        "m2": {},
        "perceptron": {},
        "cnn": {}
    }

    n_epochs = 2
    # define classification tests
    tests = [["t1", "t2"], ["t1", "t3"]]

    if True:
        # test 1 > standalone models
        # load the data
        with open("data/probabilities_processed.json", "r") as jf:
            json_data = load(jf)

        for i, sets in enumerate(tests):
            set_map = json_data["sets"]
            sets = [list(set_map.values()).index(x) for x in sets]
            data = [x for x in json_data["data"] if x[0] in sets]

            # single out each model data
            perce_data0 = [[x[0], [x[1][0]]] for x in data]
            perce_data1 = [[x[0], [x[1][1]]] for x in data]
            perce_data2 = [[x[0], [x[1][2]]] for x in data]

            # calculate cut size and perform cross-validation
            cut_size = round(len(data)/len(sets)/cross_validation)

            for j, perce_data in enumerate([perce_data0, perce_data1, perce_data2]):
                accuracies = []
                for n in range(cross_validation):
                    train, test = get_nth_train_test(n, cut_size, perce_data, sets)
                    acc, f1 = Perce().train_using(train, test, epochs=n_epochs)
                    accuracies.append(acc)
                results["m"+str(j)]["test"+str(i)] = accuracies

        save_results(results)

        # test 2 > perceptron
        # data already loaded
        for i, sets in enumerate(tests):
            set_map = json_data["sets"]
            sets = [list(set_map.values()).index(x) for x in sets]
            data = [x for x in json_data["data"] if x[0] in sets]

            # calculate cut size and perform cross-validation
            cut_size = round(len(data) / len(sets) / cross_validation)
            accuracies = []
            for n in range(cross_validation):
                train, test = get_nth_train_test(n, cut_size, data, sets)
                acc, f1 = Perceptron().train_using(train, test, epochs=n_epochs)
                accuracies.append(acc)
            results["perceptron"]["test" + str(i)] = accuracies

        save_results(results)

    # test 3 > cnn
    # load the data
    with open("data/prob_vectors_processed.json", "r") as jf:
        json_data = load(jf)

    for i, sets in enumerate(tests):
        set_map = json_data["sets"]
        sets = [list(set_map.values()).index(x) for x in sets]
        data = [x for x in json_data["data"] if x[0] in sets]

        # calculate cut size and perform cross-validation
        cut_size = round(len(data) / len(sets) / cross_validation)
        accuracies = []
        for n in range(cross_validation):
            train, test = get_nth_train_test(n, cut_size, data, sets)
            acc, f1 = CNNet().train_using(train, test, epochs=n_epochs)
            accuracies.append(acc)
        results["cnn"]["test" + str(i)] = accuracies

    save_results(results)
