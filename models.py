from torch import nn, relu, cat, sigmoid, device, cuda, load as torch_load, no_grad, optim, manual_seed
from torch import FloatTensor, LongTensor, softmax, max as torch_max, save as torch_save, Generator, log_softmax
from torch.utils.data import Dataset, DataLoader
from math import floor
from sklearn.model_selection import train_test_split
from torchmetrics import F1Score, Accuracy
from tqdm import trange
from random import seed
from numpy.random import seed as npseed


num_feature = 3
num_classes = 2
max_sequence_length = 57
n_additional_features = 57
cuda.manual_seed(0)
manual_seed(0)
seed(0)
accuracy = Accuracy(task="multiclass", num_classes=num_classes)
f1 = F1Score(task="multiclass", num_classes=num_classes)


def prepare_data(training_set, test_set, val_size):
    x_train = list(zip(*training_set))[1]
    y_train = list(zip(*training_set))[0]
    x_test = list(zip(*test_set))[1]
    y_test = list(zip(*test_set))[0]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size,
                                                      stratify=y_train, random_state=0)
    train_dataset = DatasetMapper(FloatTensor(x_train), LongTensor(y_train))
    val_dataset = DatasetMapper(FloatTensor(x_val), LongTensor(y_val))
    test_dataset = DatasetMapper(FloatTensor(x_test), LongTensor(y_test))

    return train_dataset, val_dataset, test_dataset


class DatasetMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def pairwise(lst):
    it = iter(lst)
    a = next(it, None)
    for b in it:
        yield a, b
        a = b


def seed_worker(worker_id):
    worker_seed = 0
    npseed(worker_seed)
    seed(worker_seed)


g = Generator()
g.manual_seed(0)


def pad_inputs(inputs, seq_len):
    def pad(seq):
        length = len(seq)
        if length > seq_len:
            return seq[:seq_len]
        for i in range(seq_len - length):
            pad_value = 0
            # pad_value = seq[i % length]
            seq.append(pad_value)
        return seq
    return [[pad(x) for x in y] for y in inputs]


class NN(nn.Module):
    def evaluate(self, inputs, probability=False, perceptron=False):
        if not perceptron:
            inputs = pad_inputs(inputs, self.seq_len)
        data = DatasetMapper(FloatTensor(inputs), FloatTensor([0 for _ in inputs]))
        data_loader = DataLoader(dataset=data, batch_size=1, worker_init_fn=seed_worker)
        y_prediction_list = []
        y_probability_list = []
        with no_grad():
            self.eval()
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                output = self.forward(x_batch)
                if not perceptron:
                    output = output.unsqueeze(0)
                y_probability, y_prediction = torch_max(softmax(output, dim=1), dim=1)
                y_prediction_list.append(y_prediction.cpu().numpy().squeeze().tolist())
                y_probability_list.append(y_probability.cpu().numpy().squeeze().tolist())
        if probability:
            return y_prediction_list[0], round(y_probability_list[0], 4)
        return y_prediction_list[0]


class Perceptron(NN):
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

    def train_using(self, training_set, test_set, val_size=0.01, batch=64, learning_rate=0.01, epochs=10, save_path=""):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        accuracy.to(self.device)
        f1.to(self.device)
        train_dataset, val_dataset, test_dataset = prepare_data(training_set, test_set, val_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch, worker_init_fn=seed_worker)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, worker_init_fn=seed_worker)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, worker_init_fn=seed_worker)
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


class MultiNN(NN):
    def __init__(self, model_path="",
                 stride=2, cnn_features=8, rnn_features=8, num_channels=3, dropout=0,
                 kernels=None, layers=None):
        super(MultiNN, self).__init__()
        self.device = device("cuda:0" if cuda.is_available() else "cpu")

        # Model parameters
        self.seq_len: int = max_sequence_length
        self.num_channels: int = num_channels
        self.cnn_features: int = cnn_features
        self.rnn_features: int = rnn_features
        self.stride: int = stride
        self.dropout = nn.Dropout(dropout)
        self.padding = 0

        if kernels is None:
            self.kernels = [3]
        else:
            self.kernels = kernels
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        # CNN layers definition
        self.max_pools = nn.ModuleList()
        self.convolutions = nn.ModuleList()

        for kernel in self.kernels:
            x = nn.Conv1d(self.num_channels,
                          self.cnn_features,
                          kernel_size=kernel,
                          stride=self.stride,
                          padding=self.padding
                          ).to(self.device)
            self.convolutions.append(x)
            z = nn.MaxPool1d(kernel_size=kernel, stride=self.stride).to(self.device)
            self.max_pools.append(z)

        num_cnn_features = self.cnn_out_features()
        self.cnn_fc = nn.Linear(num_cnn_features, self.cnn_features).to(self.device)
        self.add_fc = nn.Linear(n_additional_features-self.num_channels, self.cnn_features).to(self.device)

        # RNN layers definition
        self.rnn = nn.RNN(input_size=num_channels, hidden_size=self.rnn_features, dropout=dropout)

        # Fully connected layers definition
        self.fcl = nn.ModuleList()

        for a, b in pairwise([self.cnn_features*2+self.rnn_features+self.num_channels, *self.layers, num_classes]):
            self.fcl.append(nn.Linear(a, b).to(self.device))

        if model_path:
            self.load_state_dict(torch_load(model_path, map_location=device(self.device)))
        self.to(self.device)

    def forward(self, x):
        # ensure float
        x = x.float().swapaxes(0, 1)
        # separate standard and additional features
        add = x[self.num_channels:].swapaxes(0, 1).squeeze(1)
        x = x[:self.num_channels].swapaxes(0, 1)

        # cnn
        cnn_outs = []
        for i, k in enumerate(self.kernels):
            z = self.convolutions[i](x)
            z = relu(z)
            z = self.max_pools[i](z)
            cnn_outs.append(z)

        # rnn
        y = x.swapaxes(0, 2).swapaxes(1, 2)
        z = self.rnn(y)[1]
        rnn_outs = [z.squeeze(0).unsqueeze(2)]

        # flatten cnn features
        cnn_outs = cat(tuple(cnn_outs), 2)
        cnn_outs = cnn_outs.reshape(cnn_outs.size(0), -1)
        cnn_outs = self.cnn_fc(cnn_outs)

        # compile added features
        add_out = self.add_fc(add[:, :n_additional_features-self.num_channels])

        # flatten rnn features
        rnn_outs = cat(tuple(rnn_outs), 2)
        rnn_outs = rnn_outs.reshape(rnn_outs.size(0), -1)

        out = cat((cnn_outs, rnn_outs, add_out, add[:, n_additional_features-self.num_channels:]), dim=1)

        # The "flattened" vector is passed through a fully connected layers
        for fc in self.fcl:
            out = relu(out)
            out = fc(out)

        # out = log_softmax(out, dim=1)
        return out.squeeze()

    def pack_features(self, arr):
        padded_input = pad_inputs([arr], self.seq_len)[0]
        return padded_input

    def train_using(self, training_set, test_set, val_size=0.01, batch=64, learning_rate=0.01, epochs=10, save_path=""):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        accuracy.to(self.device)
        f1.to(self.device)
        training_set = [[x[0], self.pack_features(x[1])] for x in training_set]
        test_set = [[x[0], self.pack_features(x[1])] for x in test_set]
        train_dataset, val_dataset, test_dataset = prepare_data(training_set, test_set, val_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch, worker_init_fn=seed_worker)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, worker_init_fn=seed_worker)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, worker_init_fn=seed_worker)
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
        print(test_accuracy)
        return test_accuracy, test_f1

    def cnn_out_features(self):
        out_pools = []
        for i, kernel in enumerate(self.kernels):
            conv_out_size = ((self.seq_len - self.kernels[i]) / self.stride) + 1
            pool_out_size = ((floor(conv_out_size) - self.kernels[i]) / self.stride) + 1
            out_pools.append(floor(pool_out_size))

        return sum(out_pools) * self.cnn_features + self.padding * self.cnn_features
