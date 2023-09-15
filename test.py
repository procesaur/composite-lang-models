from models import Perceptron, CNNet
from json import load


test_net_outputs = False
training_and_testing = True
cross_validation = 5


def get_nth_train_test(n, cut, lst, classes):
    train_set = []
    test_set = []
    for c in classes:
        rows = [x for x in lst if x[0] == c]
        train_set.extend(rows[:n*cut])
        train_set.extend(rows[(n+1)*cut:])
        test_set.extend(rows[n*cut:(n+1)*cut])
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
    # load the data
    with open("data/probabilities_processed.json", "r") as jf:
        json_data = load(jf)

    # define classification tests
    tests = [["t1", "t2"], ["t1", "t3"]]
    for sets in tests:
        set_map = json_data["sets"]
        sets = [list(set_map.values()).index(x) for x in sets]
        data = [x for x in json_data["data"] if x[0] in sets]

        cut_size = round(len(data)/len(sets)/cross_validation)
        for n in range(cross_validation):
            train, test = get_nth_train_test(n, cut_size, json_data["data"], sets)
            acc, f1 = Perceptron().train_using(train, test)
            print(acc, f1)
