from models import Perceptron, CNNet


model_path = "D:/aplikacije_mihailo/Parallel-language-models/nets/sem-ord.pt"
model_path = "D:/aplikacije_mihailo/Parallel-language-models/nets/general_cnn.pt"
inputs = [[0, 0, 0.1]]
inputs = [[[0.1, 0.2, 0.1], [0.1], [0.9]]]

x = CNNet(model_path).evaluate(inputs=inputs, probability=True)

print(x)
