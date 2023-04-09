from time import time, strftime, gmtime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def time_counter(func):
    def wrapper(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()
        delta = te - ts
        print(f'{func.__name__} выполнялся {strftime("%H:%M:%S", gmtime(delta))}')
        return result
    return wrapper


def print_metrics_scores(scores: dict) -> None:
    print(f'confusion_matrix: \n{scores.get("confusion_matrix")}')
    print(f'classification_report: \n{scores.get("classification_report")}')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, 3, 1)             # 3*3*1*50+50 = 500
        self.conv2 = nn.Conv2d(50, 20, 3, 1)            # 3*3*50*20+20 = 9020
        self.conv3 = nn.Conv2d(20, 8, 3, 1)            # 3*3*20*8+8 = 1448
        self.dropout1 = nn.Dropout(0.25)                # <- 0
        self.dropout2 = nn.Dropout(0.5)                 # <- 0
        self.fc1 = nn.Linear(200, 128)                  # <- 200*128+128 = 25728
        self.fc2 = nn.Linear(128, 10)                   # <- 128*10+10 = 1290
        self.batch_norm = torch.nn.BatchNorm1d(200)     # <- 200*2 = 400
                                                        # Total: 38386
    def forward(self, x):           # 200, 1, 28, 28
        x = self.conv1(x)           # 200, 50, 26, 26
        x = F.relu(x)               # 200, 50, 26, 26
        x = self.conv2(x)           # 200, 20, 24, 24
        x = F.relu(x)               # 200, 20, 24, 24
        x = F.max_pool2d(x, 2)      # 200, 20, 12, 12
        x = self.conv3(x)           # 200, 8, 10, 10
        x = F.relu(x)               # 200, 8, 10, 10
        x = F.max_pool2d(x, 2)      # 200, 8, 5, 5
        x = self.dropout1(x)        # 200, 8, 5, 5
        x = torch.flatten(x, 1)     # 200, 8*5*5 = 200
        x = self.batch_norm(x)      # 200, 200
        x = self.fc1(x)             # 200, 128
        x = F.relu(x)               # 200, 128
        x = self.dropout2(x)        # 200, 128
        output = self.fc2(x)        # 200, 10
        return output


def model_train(model, device, train_loader, optimizer, epoch, loss_fn):
    train_loss = 0
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_accuracy = correct / len(train_loader.dataset)
    train_accuracy_list.append(train_accuracy)


def model_test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    all_target = []
    all_pred = []
    classification_metrics_dict = {}
    with torch.no_grad():
        for batch_idx_, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += loss_fn(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_target = [*all_target, *target.tolist()]
            all_pred = [*all_pred, *pred.tolist()]
            if batch_idx_ == len(test_loader) - 1:
                target_names = ['cls 0', 'cls 1', 'cls 2', 'cls 3', 'cls 4',
                                'cls 5', 'cls 6', 'cls 7', 'cls 8', 'cls 9']
                all_target = torch.Tensor(all_target)
                all_pred = torch.Tensor(all_pred)
                classification_metrics_dict.update({
                    'confusion_matrix': confusion_matrix(all_target, all_pred),
                    'classification_report': classification_report(all_target, all_pred, target_names=target_names)
                })
                print_metrics_scores(classification_metrics_dict)
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=classification_metrics_dict['confusion_matrix'],
                    display_labels=target_names
                )
                disp.plot()
                plt.savefig(f"Confusion_matrix.png")
                plt.close()

    test_loss /= len(test_loader.dataset)
    test_loss_list.append(test_loss)
    test_accuracy = correct / len(test_loader.dataset)
    test_accuracy_list.append(test_accuracy)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * test_accuracy))


@time_counter
def main():

    torch.manual_seed(1337)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size_ds = 200
    train_kwargs = {'batch_size': batch_size_ds}
    test_kwargs = {'batch_size': batch_size_ds}

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST('./mnsit-dataset', train=True, download=True,
                              transform=transform)
    test_dataset = datasets.MNIST('./mnsit-dataset', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    epochs = 5

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    for epoch in range(epochs):
        model_train(model, device, train_loader, optimizer, epoch, loss_fn)
        model_test(model, device, test_loader, loss_fn)
        scheduler.step()

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)
    print(summary(model, input_size=(1, 28, 28)))


def plot_curve(title_: str, xlabel_: str, ylabel_: str,
               curve_list_1: list, curve_list_2: list, curve_label_1: str,
               curve_label_2: str, file_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.title(f'{title_}')
    plt.xlabel(f'{xlabel_}')
    plt.ylabel(f'{ylabel_}')
    ax.plot(np.arange(0, len(curve_list_1), 1), curve_list_1, "o-", label=curve_label_1)
    ax.plot(np.arange(0, len(curve_list_2), 1), curve_list_2, "o-", label=curve_label_2)
    ax.legend(loc="best")

    plt.savefig(f"{file_name}")
    plt.close()


if __name__ == '__main__':
    # Init lists
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    main()

    # Plot results
    plot_curve('Loss function', 'Samples', 'Loss',
               train_loss_list, test_loss_list,
               "Train loss", "Test loss", "Loss.png")
    plot_curve('Loss function', 'Samples', 'Loss',
               train_loss_list, test_loss_list,
               "Log train loss", "Log test loss", "Loss_log.png")
    plot_curve('Accuracy curve', 'Epoch', 'Accuracy',
               train_accuracy_list, test_accuracy_list,
               "Train", "Test", "Accuracy curve.png")
