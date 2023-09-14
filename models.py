from torch import nn, relu, cat, sigmoid, device, cuda, load, no_grad, FloatTensor, softmax, max as torch_max
from torch.utils.data import Dataset, DataLoader
from math import floor


batch = 64
num_feature = 3
num_classes = 2
max_sequence_length = 128


class DatasetMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Perceptron(nn.Module):
    def __init__(self, model_path=""):
        super(Perceptron, self).__init__()
        self.layer_out = nn.Linear(num_feature, num_classes)
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        if model_path:
            self.load_state_dict(load(model_path, map_location=device(self.device)))
            self.to(self.device)

    def forward(self, x):
        x = self.layer_out(x)
        return x

    def evaluate(self, inputs, probability=False):
        data = DatasetMapper(FloatTensor(inputs), FloatTensor([0 for _ in inputs]))
        data_loader = DataLoader(dataset=data, batch_size=1)
        y_prediction_list = []
        y_probability_list = []
        with no_grad():
            self.eval()
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                y_probability, y_prediction = torch_max(softmax(self.forward(x_batch), dim=1), dim=1)
                y_prediction_list.append(y_prediction.cpu().numpy().squeeze().tolist())
                y_probability_list.append(y_probability.cpu().numpy().squeeze().tolist())
        if probability:
            return y_prediction_list[0], round(y_probability_list[0], 4)
        return y_prediction_list[0]


class CNNet(nn.ModuleList):
    def __init__(self, model_path=""):
        super(CNNet, self).__init__()
        self.seq_len: int = 3
        # Model parameters
        self.embedding_size: int = max_sequence_length
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.out_size: int = 32
        self.stride: int = 3
        self.kernels = [4, 5]
        self.dropout = nn.Dropout(0.25)

        # CNN parameters definition
        self.convolutions = []
        self.pools = []
        for kernel in self.kernels:
            # Convolution layers definition
            x = nn.Conv1d(self.seq_len, self.out_size, tuple([kernel]), tuple([self.stride])).to(self.device)
            self.convolutions.append(x)
            y = nn.MaxPool1d(kernel, self.stride).to(self.device)
            self.pools.append(y)

        # Fully connected layer definition
        self.fc = nn.Linear(self.in_features_fc(), 2)
        if model_path:
            self.load_state_dict(load(model_path, map_location=device(self.device)))
            self.to(self.device)

    def pad_inputs(self, inputs):
        def pad(seq):
            length = len(seq)
            if length > self.embedding_size:
                return seq[:self.embedding_size]
            for i in range(self.embedding_size - length):
                seq.append(seq[i % length])
            return seq
        return [[pad(x) for x in y] for y in inputs]

    def in_features_fc(self):
        out_pools = []
        for i, kernel in enumerate(self.kernels):
            x = ((self.embedding_size - 1 * (self.kernels[i] - 1) - 1) / self.stride) + 1
            y = ((floor(x) - 1 * (self.kernels[i] - 1) - 1) / self.stride) + 1
            out_pools.append(floor(y))

        # Returns "flattened" vector (input for fully connected layer)
        return (sum(out_pools)) * self.out_size

    def forward(self, x):
        x = x.float()
        outs = []
        for i, k in enumerate(self.kernels):
            y = self.convolutions[i](x)
            y = relu(y)
            y = self.pools[i](y)
            outs.append(y)

        union = cat(tuple(outs), 2)
        union = union.reshape(union.size(0), -1)
        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(union)
        out = self.dropout(out)
        out = sigmoid(out)

        return out.squeeze()

    def evaluate(self, inputs, probability=False):
        data = DatasetMapper(FloatTensor(self.pad_inputs(inputs)), FloatTensor([0 for _ in inputs]))
        data_loader = DataLoader(dataset=data, batch_size=1)
        y_prediction_list = []
        y_probability_list = []
        with no_grad():
            self.eval()
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                y_probability, y_prediction = torch_max(softmax(self.forward(x_batch).unsqueeze(0), dim=1), dim=1)
                y_prediction_list.append(y_prediction.cpu().numpy().squeeze().tolist())
                y_probability_list.append(y_probability.cpu().numpy().squeeze().tolist())
        if probability:
            return y_prediction_list[0], round(y_probability_list[0], 4)
        return y_prediction_list[0]
