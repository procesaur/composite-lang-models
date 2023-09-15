from models import Perceptron, CNNet
from json import load


test_net_outputs = False
training_and_testing = True

if test_net_outputs:
    model_path = "D:/aplikacije_mihailo/Parallel-language-models/nets/sem-ord.pt"
    inputs = [[0, 0.1, 0.1]]
    x = Perceptron(model_path).evaluate(inputs, True)
    print(x)

    model_path = "D:/aplikacije_mihailo/Parallel-language-models/nets/general_cnn.pt"
    inputs = [[[0.1, 0.2, 0.1], [0.1], [0.9]]]
    x = CNNet(model_path).evaluate(inputs, True)
    print(x)


if training_and_testing:
    # load the data
    with open("data/probabilities_processed.json", "r") as jf:
        data = load(jf)

    # define tests
    tests = [["t1, t2"], ["t1", "t3"]]
    tests = [["t1", "t2"]]
    for sets in tests:
        set_map = data["sets"]
        sets = [list(set_map.values()).index(x) for x in sets]
        data = [x for x in data["data"] if x[0] in sets]






    x = Perceptron().train_using(data)

# TO DO > CONFIGURE DATA LOADING