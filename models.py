from torch import nn, relu, cat, sigmoid, device, cuda, load as torch_load, no_grad, optim, manual_seed
from torch import FloatTensor, LongTensor, softmax, max as torch_max, save as torch_save
from torch.utils.data import Dataset, DataLoader
from math import floor
from sklearn.model_selection import train_test_split
from torchmetrics import F1Score, Accuracy
from tqdm import trange
from random import seed


num_feature = 3
num_classes = 2
max_sequence_length = 64
manual_seed(0)
seed(0)
accuracy = Accuracy(task="multiclass", num_classes=num_classes)
f1 = F1Score(task="multiclass", num_classes=num_classes)


class DatasetMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def prepare_data(training_set, test_set, val_size):
    x_train = list(zip(*training_set))[1]
    y_train = list(zip(*training_set))[0]
    x_test = list(zip(*test_set))[1]
    y_test = list(zip(*test_set))[0]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size,
                                                      stratify=y_train, random_state=12)
    train_dataset = DatasetMapper(FloatTensor(x_train), LongTensor(y_train))
    val_dataset = DatasetMapper(FloatTensor(x_val), LongTensor(y_val))
    test_dataset = DatasetMapper(FloatTensor(x_test), LongTensor(y_test))

    return train_dataset, val_dataset, test_dataset


class Perceptron(nn.Module):
    def __init__(self, model_path=""):
        super(Perceptron, self).__init__()
        self.layer_out = nn.Linear(num_feature, num_classes)
        if model_path:
            self.load_state_dict(torch_load(model_path, map_location=device(self.device)))
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
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

    def train_using(self, training_set, test_set, val_size=0.1, batch=64, learning_rate=0.01, epochs=10, save_path=""):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        accuracy.to(self.device)
        f1.to(self.device)
        train_dataset, val_dataset, test_dataset = prepare_data(training_set, test_set, val_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)
        best = 0

        epochs = trange(1, epochs)
        for _ in epochs:
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            self.train()
            for x_train_batch, y_train_batch in train_loader:
                x_train_batch, y_train_batch = x_train_batch.to(self.device), y_train_batch.to(self.device)
                optimizer.zero_grad()

                y_train_predictions = self(x_train_batch)
                train_loss = criterion(y_train_predictions, y_train_batch)
                train_acc = accuracy(y_train_predictions, y_train_batch).item()
                train_loss.backward()

                optimizer.step()
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc

            train_loss = train_epoch_loss / len(train_loader)
            train_acc = train_epoch_acc / len(train_loader)

            # VALIDATION
            y_val_predictions = []
            y_validation = []
            with no_grad():

                self.eval()

                for x_val_batch, y_val_batch in val_loader:
                    x_val_batch, y_val_batch = x_val_batch.to(self.device), y_val_batch.to(self.device)

                    y_val_predictions.append(self(x_val_batch))
                    y_validation.append(y_val_batch)

            val_loss = criterion(cat(y_val_predictions), cat(y_validation)).item()
            val_acc = accuracy(cat(y_val_predictions), cat(y_validation)).item()

            epochs.set_description(str(round(train_loss, 3)) + "/" + str(round(val_loss, 3)) + "/"
                                   + str(round(train_acc, 3)) + "/" + str(round(val_acc, 3)))
            epochs.update()
            if val_acc > best:
                best = val_acc
                if save_path:
                    torch_save(self.state_dict(), save_path)

        y_prediction_list = []
        y_validation = []
        with no_grad():
            self.eval()
            for x_test_batch, y_test_batch in test_loader:
                x_test_batch, y_test_batch = x_test_batch.to(self.device), y_test_batch.to(self.device)
                y_prediction_list.append(self(x_test_batch))
                y_validation.append(y_test_batch)

        test_accuracy = accuracy(cat(y_prediction_list), cat(y_validation)).item()
        test_f1 = f1(cat(y_prediction_list), cat(y_validation)).item()

        return test_accuracy, test_f1


class Perce(Perceptron):
    def __init__(self, model_path=""):
        super(Perceptron, self).__init__()
        self.layer_out = nn.Linear(1, num_classes)
        if model_path:
            self.load_state_dict(torch_load(model_path, map_location=device(self.device)))
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.to(self.device)


class CNNet(nn.ModuleList):
    def __init__(self, model_path="", stride=2, out_size=32, seq_length=3, dropout=0.25, kernels=None):
        super(CNNet, self).__init__()
        self.seq_len: int = seq_length
        # Model parameters
        self.embedding_size: int = max_sequence_length
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.out_size: int = out_size
        self.stride: int = stride
        self.dropout = nn.Dropout(dropout)
        if kernels is None:
            self.kernels = [3, 5]
        else:
            self.kernels = kernels

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
        self.fc = nn.Linear(self.in_features_fc(), 32)
        self.fc2 = nn.Linear(32, 2)
        if model_path:
            self.load_state_dict(torch_load(model_path, map_location=device(self.device)))
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
        out = relu(out)
        out = self.fc2(out)
        # out = self.dropout(out)
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

    def train_using(self, training_set, test_set, val_size=0.1, batch=64, learning_rate=0.01, epochs=10, save_path=""):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        accuracy.to(self.device)
        f1.to(self.device)
        training_set = [[x[0], self.pad_inputs([x[1]])[0]] for x in training_set]
        test_set = [[x[0], self.pad_inputs([x[1]])[0]] for x in test_set]
        train_dataset, val_dataset, test_dataset = prepare_data(training_set, test_set, val_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)
        best = 0

        epochs = trange(1, epochs)
        for _ in epochs:
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            self.train()
            for x_train_batch, y_train_batch in train_loader:
                x_train_batch, y_train_batch = x_train_batch.to(self.device), y_train_batch.to(self.device)
                optimizer.zero_grad()

                y_train_predictions = self(x_train_batch)
                train_loss = criterion(y_train_predictions, y_train_batch)
                train_acc = accuracy(y_train_predictions, y_train_batch)
                train_loss.backward()

                optimizer.step()
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            train_loss = train_epoch_loss / len(train_loader)
            train_acc = train_epoch_acc / len(train_loader)

            # VALIDATION
            y_val_predictions = []
            y_validation = []
            with no_grad():

                self.eval()
                for x_val_batch, y_val_batch in val_loader:
                    x_val_batch, y_val_batch = x_val_batch.to(self.device), y_val_batch.to(self.device)

                    y_val_predictions.append(self(x_val_batch).unsqueeze(0))
                    y_validation.append(y_val_batch)

            val_loss = criterion(cat(y_val_predictions), cat(y_validation)).item()
            val_acc = accuracy(cat(y_val_predictions), cat(y_validation)).item()

            epochs.set_description(str(round(train_loss, 3)) + "/" + str(round(val_loss, 3)) + "/"
                                   + str(round(train_acc, 3)) + "/" + str(round(val_acc, 3)))
            epochs.update()
            if val_acc > best:
                best = val_acc
                if save_path:
                    torch_save(self.state_dict(), save_path)

        y_prediction_list = []
        y_validation = []
        with no_grad():
            self.eval()
            for x_test_batch, y_test_batch in test_loader:
                x_test_batch, y_test_batch = x_test_batch.to(self.device), y_test_batch.to(self.device)
                y_prediction_list.append(self(x_test_batch).unsqueeze(0))
                y_validation.append(y_test_batch)

        test_accuracy = accuracy(cat(y_prediction_list), cat(y_validation)).item()
        test_f1 = f1(cat(y_prediction_list), cat(y_validation)).item()

        return test_accuracy, test_f1
