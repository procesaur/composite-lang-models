from torch import flatten, max as torch_max, log_softmax, sigmoid, abs, no_grad, save, device as device_x,\
    cuda, nn, optim, tensor, from_numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange
from json import load as j_load

import numpy as np
from models import DatasetMapper, DataLoader, CNNet, Perceptron, batch


device = device_x("cuda:0" if cuda.is_available() else "cpu")


def separability(l1, l2):
    total = len(l1) + len(l2)
    mins = min(l2)
    maxs = max(l1)
    l1.sort()
    l2.sort()
    i1 = [x for x in l2 if x <= maxs]
    i2 = [x for x in l1 if x > mins]
    cand = set(i1).union(set(i2))
    results = []
    for c in cand:
        misses = fitness(l1, l2, c)
        acc = misses / total
        results.append(acc)
    return max(results)


def fitness(l1, l2, z):
    z += 0.000001
    j1 = [x for x in l2 if x < z]
    j2 = [x for x in l1 if x > z]
    misses = len(j1) + len(j2)
    return misses


def get_epochs_n(n):
    epochs = round(1500000 / n) + 10
    # print(epochs)
    return epochs


def multi_acc(y_pred, y_test):
    y_pred_softmax = log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch_max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = round(acc) * 100

    return acc


def get_class_distribution(obj):
    count_dict = {0: 0, 1: 1}
    for i in obj:
        count_dict[i] += 1
    return count_dict


def get_sep(inputs, outputs):
    # get two arrays of x for test
    a = []
    b = []
    for x, y in zip(inputs, outputs):
        if y == 0:
            a.append(x[0])
        else:
            b.append(x[0])
    print(round(separability(a, b), 4))


def custom_acc(y_pred, y_batch):
    pred = flatten(y_pred)
    pred = sigmoid(pred)
    x = abs(y_batch - pred)
    correct = len([y for y in x if y < 0.5])
    accu = 100 * correct / x.size(dim=0)
    return accu


def for_crosvalid(inputs, outputs, c=5):
    chunks = []
    nnn = len(inputs)
    n = round(nnn / c)
    for i in range(c):
        try:
            chunks.append({"inputs": inputs[i * n:(i + 1) * n], "outputs": outputs[i * n:(i + 1) * n]})
        except:
            chunks.append({"inputs": inputs[i * n:], "outputs": outputs[i * n:]})
    sets = []
    for i, chunk in enumerate(chunks):
        test = chunk
        train = {"inputs": [], "outputs": []}
        for j, _ in enumerate(chunks):
            if i != j or c < 2:
                train["inputs"] += chunks[j]["inputs"]
                train["outputs"] += chunks[j]["outputs"]
        sets.append({"train": train, "test": test})
    return sets


def prepare_dataset(path, test, cnn=False):
    with open(path, "r", encoding="utf8") as j:
        rj = j_load(j)
    inputs = rj[test]["inputs"]
    outputs = rj[test]["outputs"]
    return inputs, outputs


def train(batch_size=batch, lr=0.01, val_size=0.1, cross=1, savepath="", datapath="data/for_training.json",
               test="form-ord", hard=False, cnn=False):
    crit = nn.CrossEntropyLoss()
    # fill inputs and outputs
    inputs, outputs = prepare_dataset(datapath, test, cnn)
    # set epochs
    epochs = get_epochs_n(len(outputs))
    # get crossvalidation sets
    sets = for_crosvalid(inputs, outputs, cross)

    f1s = []
    accuracy = []
    for s in sets:
        # create train, test and val sets
        X_trainval = s["train"]["inputs"]
        X_test = s["test"]["inputs"]
        y_trainval = s["train"]["outputs"]
        y_test = s["test"]["outputs"]

        if hard:
            X_trainval, y_trainval, X_test, y_test = X_test, y_test, X_trainval, y_trainval

        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size,
                                                          stratify=y_trainval, random_state=12)

        # convert them to np arrays
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = np.array(X_val), np.array(y_val)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # convert them to classiferdatasets
        if cnn:
            dataset = DatasetMapper
        else:
            dataset = DatasetMapper

        train_dataset = dataset(tensor(X_train).float(), from_numpy(y_train).long())
        val_dataset = dataset(from_numpy(X_val).float(), from_numpy(y_val).long())
        test_dataset = dataset(from_numpy(X_test).float(), from_numpy(y_test).long())

        # initiate loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        # initiate model and move it to device

        if cnn:
            model = CNNet()
        else:
            model = Perceptron(3)
        model.to(device)

        # initiate optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        accuracy_stats = {
            'train': [],
            "val": [],
        }
        loss_stats = {
            'train': [],
            "val": [],
        }

        # print("Begin training.")
        best = 0

        tt = trange(1, epochs)
        for e in tt:

            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0

            model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer.zero_grad()

                y_train_pred = model(X_train_batch)

                train_loss = crit(y_train_pred, y_train_batch)
                train_acc = multi_acc(y_train_pred, y_train_batch)
                train_loss.backward()

                optimizer.step()
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            # VALIDATION
            with no_grad():

                val_epoch_loss = 0
                val_epoch_acc = 0

                model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                    y_val_pred = model(X_val_batch)
                    if cnn:
                        y_val_pred = y_val_pred.unsqueeze(0)
                    val_loss = crit(y_val_pred, y_val_batch)
                    val_acc = multi_acc(y_val_pred, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss / len(train_loader))
            loss_stats['val'].append(val_epoch_loss / len(val_loader))
            accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

            tloss = train_epoch_loss / len(train_loader)
            vloss = val_epoch_loss / len(val_loader)
            tacc = train_epoch_acc / len(train_loader)
            vacc = val_epoch_acc / len(val_loader)

            # print(f'Epoch {e + 0:03}: |'f' Train Loss: {tloss:.5f} |'f' Val Loss: {vloss:.5f} |'f' Train Acc: {tacc:.3f}|'f' Val Acc: {vacc:.3f}')

            score = vacc
            tt.set_description(str(round(tacc, 3)) + "/" + str(round(vacc,3)))
            tt.update()
            if score > best:
                best = score
                if savepath:
                    save(model.state_dict(), savepath)

        # print(model.layer_out.weight)
        y_pred_list = []

        with no_grad():
            model.eval()
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                y_test_pred = model(X_batch)
                if cnn:
                    y_test_pred = y_test_pred.unsqueeze(0)
                y_pred_softmax = log_softmax(y_test_pred, dim = 1)
                _, y_pred_tags = torch_max(y_pred_softmax, dim = 1)
                y_pred_list.append(y_pred_tags.cpu().numpy())

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        # report = classification_report(y_test, y_pred_list, digits=4)
        f1s.append(f1_score(y_test, y_pred_list, average="macro"))
        ta = accuracy_score(y_test, y_pred_list, normalize=True, sample_weight=None)
        print(ta)
        accuracy.append(ta)

    for ass, f1ss in zip(accuracy, f1s):
        print(ass, f1ss)