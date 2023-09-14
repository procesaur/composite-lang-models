from torch import nn, relu, cat, sigmoid, device, cuda, load as torch_load, no_grad, optim
from torch import FloatTensor, LongTensor, softmax, max as torch_max, save as torch_save, log_softmax
from torch.utils.data import Dataset, DataLoader
from math import floor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange


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


def multi_acc(y_predictions, y_test):
    _, y_prediction_tags = torch_max(log_softmax(y_predictions, dim=1), dim=1)
    correct_predictions = (y_prediction_tags == y_test).float()
    acc = correct_predictions.sum() / len(correct_predictions)
    acc = round(acc) * 100
    return acc


def prepare_data(data, val_size):
    classes = {cls: i for i, cls in enumerate(data["train"])}
    training_set = [(classes[cls], x for x in data["train"][cls]) for cls in data["train"]]
    test_set = [(classes[cls], x for x in data["test"][cls]) for cls in data["test"]]
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
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        if model_path:
            self.load_state_dict(torch_load(model_path, map_location=device(self.device)))
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

    def training(self, data, val_size=0.1, batch_size=64, learning_rate=0.01, epochs=10, save_path=""):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_dataset, val_dataset, test_dataset = prepare_data(data, val_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        accuracy_stats = {'train': [], "val": []}
        loss_stats = {'train': [], "val": []}
        best = 0

        epochs = trange(1, epochs)
        for _ in epochs:
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            self.train()
            for x_train_batch, y_train_batch in train_loader:
                x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(self.device)
                optimizer.zero_grad()

                y_train_predictions = self(x_train_batch)
                train_loss = criterion(y_train_predictions, y_train_batch)
                train_acc = multi_acc(y_train_predictions, y_train_batch)
                train_loss.backward()

                optimizer.step()
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc

            # VALIDATION
            with no_grad():

                val_epoch_loss = 0
                val_epoch_acc = 0

                self.eval()
                for x_val_batch, y_val_batch in val_loader:
                    x_val_batch, y_val_batch = x_val_batch.to(self.device), y_val_batch.to(self.device)

                    y_val_predictions = self(x_val_batch)
                    val_loss = criterion(y_val_predictions, y_val_batch)
                    val_acc = multi_acc(y_val_predictions, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc

            loss_stats['train'].append(train_epoch_loss / len(train_loader))
            loss_stats['val'].append(val_epoch_loss / len(val_loader))
            accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

            train_loss = train_epoch_loss / len(train_loader)
            val_loss = val_epoch_loss / len(val_loader)
            train_acc = train_epoch_acc / len(train_loader)
            val_acc = val_epoch_acc / len(val_loader)

            score = val_acc
            epochs.set_description(str(round(train_loss, 3)) + "/" + str(round(val_loss, 3)) + "/"
                                   + str(round(train_acc, 3)) + "/" + str(round(val_acc, 3)))
            epochs.update()
            if score > best:
                best = score
                if save_path:
                    torch_save(self.state_dict(), save_path)

        y_prediction_list = []
        y_validation = []
        with no_grad():
            self.eval()
            for x_val_batch, y_val_batch in test_loader:
                x_val_batch, y_val_batch = x_val_batch.to(self.device), y_val_batch.to(self.device)
                y_prediction_list.append(self(x_val_batch))
                y_validation.append(y_val_batch)

        test_accuracy = accuracy_score(y_validation, y_prediction_list, normalize=True, sample_weight=None)
        test_f1 = f1_score(y_validation, y_prediction_list, average="macro")

        print("acc: " + str(test_accuracy) + " , f1: " + str(test_f1))


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

    def training(self, data, val_size=0.1, batch_size=64, learning_rate=0.01, epochs=10, save_path=""):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_dataset, val_dataset, test_dataset = prepare_data(data, val_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        accuracy_stats = {'train': [], "val": []}
        loss_stats = {'train': [], "val": []}
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
                train_acc = multi_acc(y_train_predictions, y_train_batch)
                train_loss.backward()

                optimizer.step()
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc

            # VALIDATION
            with no_grad():

                val_epoch_loss = 0
                val_epoch_acc = 0

                self.eval()
                for x_val_batch, y_val_batch in val_loader:
                    x_val_batch, y_val_batch = x_val_batch.to(self.device), y_val_batch.to(self.device)

                    y_val_predictions = self(x_val_batch).unsqueeze(0)
                    val_loss = criterion(y_val_predictions, y_val_batch)
                    val_acc = multi_acc(y_val_predictions, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc

            loss_stats['train'].append(train_epoch_loss / len(train_loader))
            loss_stats['val'].append(val_epoch_loss / len(val_loader))
            accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

            train_loss = train_epoch_loss / len(train_loader)
            val_loss = val_epoch_loss / len(val_loader)
            train_acc = train_epoch_acc / len(train_loader)
            val_acc = val_epoch_acc / len(val_loader)

            score = val_acc
            epochs.set_description(str(round(train_loss, 3)) + "/" + str(round(val_loss, 3)) + "/"
                                   + str(round(train_acc, 3)) + "/" + str(round(val_acc, 3)))
            epochs.update()
            if score > best:
                best = score
                if save_path:
                    torch_save(self.state_dict(), save_path)

        y_prediction_list = []
        y_validation = []
        with no_grad():
            self.eval()
            for x_val_batch, y_val_batch in test_loader:
                x_val_batch, y_val_batch = x_val_batch.to(self.device), y_val_batch.to(self.device)
                y_prediction_list.append(self(x_val_batch))
                y_validation.append(y_val_batch)

        test_accuracy = accuracy_score(y_validation, y_prediction_list, normalize=True, sample_weight=None)
        test_f1 = f1_score(y_validation, y_prediction_list, average="macro")

        print("acc: " + str(test_accuracy) + " , f1: " + str(test_f1))
