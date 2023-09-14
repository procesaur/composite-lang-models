from models import Perceptron, CNNet
from json import load

if False:
    model_path = "D:/aplikacije_mihailo/Parallel-language-models/nets/sem-ord.pt"
    inputs = [[0, 0.1, 0.1]]
    x = Perceptron(model_path).evaluate(inputs, True)
    print(x)

if False:
    model_path = "D:/aplikacije_mihailo/Parallel-language-models/nets/general_cnn.pt"
    inputs = [[[0.1, 0.2, 0.1], [0.1], [0.9]]]
    x = CNNet(model_path).evaluate(inputs, True)
    print(x)

if Train:

    with open("data/probabilities.json", "r") as jf:
        probs = load(jf)

    data = {"train": probs["procesaur/gpt2-srlat"], "test": probs["procesaur/gpt2-srlat-sem"]}

    x = Perceptron().train_using(data)

# TO DO > CONFIGURE DATA LOADING