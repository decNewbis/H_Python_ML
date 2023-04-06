"""
Dataset references:
http://pjreddie.com/media/files/mnist_train.csv
http://pjreddie.com/media/files/mnist_test.csv
"""


from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


torch.manual_seed(1337)


class MnistMlp(torch.nn.Module):
    
    def __init__(self, inputnodes: int, hiddennodes: int, outputnodes: int) -> None:
        super().__init__()

        # number of nodes (neurons) in input, hidden, and output layer
        self.wih = torch.nn.Linear(in_features=inputnodes, out_features=hiddennodes)
        # self.whh = torch.nn.Linear(in_features=hiddennodes, out_features=hiddennodes)
        # self.whh_2 = torch.nn.Linear(in_features=hiddennodes, out_features=23)
        # self.who = torch.nn.Linear(in_features=23, out_features=outputnodes)
        self.who = torch.nn.Linear(in_features=hiddennodes, out_features=outputnodes)
        self.batch_norm = torch.nn.BatchNorm1d(hiddennodes)
        # self.activation = torch.nn.Sigmoid()
        # self.activation = torch.nn.Tanh()
        # self.activation = torch.nn.ReLU()
        # self.activation = torch.nn.SiLU()
        self.activation = torch.nn.GELU()
        # self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.wih(x)
        out = self.batch_norm(out)
        # out = self.activation(out)
        # out = self.whh(out)
        # out = self.activation(out)
        # out = self.whh(out)
        # out = self.dropout(out)
        # out = self.whh(out)
        # out = self.activation(out)
        # out = self.whh(out)
        # out = self.dropout(out)
        # out = self.activation(out)
        # out = self.whh_2(out)
        out = self.activation(out)
        # out = self.dropout(out)
        out = self.who(out)
        return out


class MnistDataset(Dataset):
    
    def __init__(self, filepath: Path) -> None:
        super().__init__()

        self.data_list = None
        with open(filepath, "r") as f:
            self.data_list = f.readlines()

        # conver string data to torch Tensor data type
        self.features = []
        self.targets = []
        for record in self.data_list:
            all_values = record.split(",")
            features = np.asfarray(all_values[1:])
            target = int(all_values[0])
            self.features.append(features)
            self.targets.append(target)

        self.features = torch.tensor(np.array(self.features), dtype=torch.float) / 255.0
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.long)
        # print(self.features.shape)
        # print(self.targets.shape)
        # print(self.features.max(), self.features.min())

    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def model_eval(test_loader_):
    # ##### Testing! #####

    # Switch [test\train] mode
    model.eval()
    test_loss = 0
    correct = 0
    classification_metrics_dict = {}
    with torch.inference_mode():
        all_target = []
        all_pred = []
        for batch_idx_, (features, target) in enumerate(test_loader_):
            features, target = features.to(device), target.to(device)
            output = model(features)
            current_test_loss = criterion(output, target)
            test_loss += current_test_loss.item()  # sum up batch loss
            # test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_target = [*all_target, *target.tolist()]
            all_pred = [*all_pred, *pred.tolist()]
            if batch_idx_ == len(test_loader_) - 1:
                target_names = ['cls 0', 'cls 1', 'cls 2', 'cls 3', 'cls 4',
                                'cls 5', 'cls 6', 'cls 7', 'cls 8', 'cls 9']
                all_target = torch.Tensor(all_target)
                all_pred = torch.Tensor(all_pred)
                classification_metrics_dict.update({
                    'confusion_matrix': confusion_matrix(all_target, all_pred),
                    'accuracy_score': accuracy_score(all_target, all_pred),
                    'precision_score_micro': precision_score(all_target, all_pred, average="micro"),
                    'precision_score_macro': precision_score(all_target, all_pred, average="macro"),
                    'recall_score_micro': recall_score(all_target, all_pred, average="micro"),
                    'recall_score_macro': recall_score(all_target, all_pred, average="macro"),
                    'f1_score_micro': f1_score(all_target, all_pred, average="micro"),
                    'f1_score_macro': f1_score(all_target, all_pred, average="macro"),
                    'classification_report': classification_report(all_target, all_pred, target_names=target_names)
                })
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=classification_metrics_dict['confusion_matrix'],
                    display_labels=target_names
                )
                disp.plot()
                plt.savefig(f"#{experiment_num} confusion_matrix.png")
                plt.close()

    test_loss /= len(test_loader_.dataset)
    return test_loss, correct, classification_metrics_dict


if __name__ == "__main__":
    # Plot title param
    experiment_num = 'HW_9'

    # Device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # NN architecture:
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    # hidden_nodes_2 = 23
    output_nodes = 10

    # learning rate is 0.1
    learning_rate = 0.1
    # batch size
    batch_size = 16
    # number of epochs
    epochs = 8

    # Load mnist training and testing data CSV file into a datasets
    train_dataset = MnistDataset(filepath="./mnist_train.csv")
    test_dataset = MnistDataset(filepath="./mnist_test.csv")

    # Make data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


    # Define NN
    model = MnistMlp(inputnodes=input_nodes, 
                     hiddennodes=hidden_nodes, 
                     outputnodes=output_nodes)
    # Number of parameters in the model
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device=device)
    # Define Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Init lists
    train_loss_list = []
    test_loss_list = []

    for epoch in range(epochs):
        current_train_loss = 0

        # ##### Training! #####

        model.train()
        for batch_idx, (features, target) in enumerate(train_loader):
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            current_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(features), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if batch_idx != 0:
                    train_loss_list.append(current_train_loss / batch_idx)
                else:
                    train_loss_list.append(current_train_loss)

                test_loss_list.append(model_eval(test_loader)[0])
                # Switch [test\train] mode
                model.train()

    test_result = model_eval(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_result[0], test_result[1], len(test_loader.dataset),
        100. * test_result[1] / len(test_loader.dataset)))

    # HW_9 Classification metrics
    print(f'confusion_matrix: \n{test_result[2].get("confusion_matrix")}')
    print(f'accuracy_score: {test_result[2].get("accuracy_score")}')
    print(f'precision_score [micro]: {test_result[2].get("precision_score_micro")}')
    print(f'precision_score [macro]: {test_result[2].get("precision_score_macro")}')
    print(f'recall_score [micro]: {test_result[2].get("recall_score_micro")}')
    print(f'recall_score [macro]: {test_result[2].get("recall_score_macro")}')
    print(f'f1_score [micro]: {test_result[2].get("f1_score_micro")}')
    print(f'f1_score [macro]: {test_result[2].get("f1_score_macro")}')
    print(f'classification_report: \n{test_result[2].get("classification_report")}')

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title(f'Loss function [Experiment #{experiment_num}]')
    plt.xlabel('Samples')
    plt.ylabel('Loss')
    ax.plot(np.arange(0, len(train_loss_list), 1), train_loss_list, "o-", label="Train loss")
    ax.plot(np.arange(0, len(test_loss_list), 1), test_loss_list, "o-", label="Test loss")
    ax.legend(loc="best")

    plt.savefig(f"#{experiment_num} Loss.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title(f'Loss function [Experiment #{experiment_num}]')
    plt.xlabel('Samples')
    plt.ylabel('Loss')
    ax.plot(np.arange(0, len(train_loss_list), 1), np.log(train_loss_list), "--.", label="Log train loss")
    ax.plot(np.arange(0, len(test_loss_list), 1), np.log(test_loss_list), "--.", label="Log test loss")
    ax.legend(loc="best")

    plt.savefig(f"#{experiment_num} Loss_log.png")
    plt.close()

    ##### Save Model! #####
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), "mnist_001.pth")

#######
# geometric pyramid rule
#######
# r = 4.28
# k_1 = sqrt(m * n) = 88
#######
#######
# r = 4.28
# k_1 = m * r ** 2 = 10 * 4.28 ** 2 = 183.184
# k_2 = m * r = 10* 2.28 = 22.8
#######
