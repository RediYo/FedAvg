import argparse
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from contact_dataset import ContactDataset

parser = argparse.ArgumentParser(description='Pytorch FedAvg')
parser.add_argument('--identity_code', type=str, help='identity_code for dataset')
args = parser.parse_args()
identity_code = args.identity_code
# ----------------------- LSTM -----------------------------
# 实例化对象
train_data = ContactDataset(identity_code)
print(f"train_data:{train_data}")
# 将数据集导入DataLoader，进行shuffle以及选取batch_size
# Windows里num_works只能为0，其他值会报错
train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
test_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)


INPUT_DIM = 2
OUTPUT_DIM = 3
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, input):
        print(f"input:{input}")
        lstm_out, _ = self.lstm(torch.tensor(input, dtype=torch.float32))
        print(f"lstm_out:{lstm_out[-1]}")
        tag_space = self.hidden2tag(lstm_out[-1])
        print(f"tag_space:{tag_space}")
        tag_scores = F.softmax(tag_space, dim=0)
        print(f"tag_scores:{tag_scores}")
        return tag_scores


net = LSTMTagger(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = training_data[0][0]
#     tag_scores = model(inputs)
#     print(f"pre_test: {tag_scores}")


def train(model, train_loader, epochs):
    """Train the network on the training set."""
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    total_loss = 0
    # writer = SummaryWriter('./data_log/train')
    for epoch in range(epochs):
        loss_epoch = 0
        for features, tags in train_loader:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Run our forward pass.
            tag_scores = model(features)

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, torch.tensor(tags, dtype=torch.float32))
            loss_epoch += loss
            # print("loss:", loss)
            loss.backward()
            optimizer.step()
        # writer.add_scalar("loss", loss_epoch, epoch)
        print(f"loss_epoch:{loss_epoch}")
        total_loss += loss_epoch
    return total_loss


def test(model, test_loader):
    """Validate the network on the entire test set."""
    loss_function = nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for features, tags in test_loader:
            # print(f"features:{features} - tags:{tags}")
            tag_scores = model(features)
            loss += loss_function(tag_scores, torch.tensor(tags, dtype=torch.float32))
            predicted = tag_scores
            total += 1
            pred_y = predicted.numpy()
            label_y = torch.stack(tags).numpy()
            diff = abs(np.array(pred_y - label_y))
            if np.max(diff) <= 0.1:  # 输出概率差值全部小于0.05则认为是预测正确
                correct += 1
            # if pred_y.all == label_y.all:
            #     correct += 1
            print(f"pred_y: {pred_y}")
            print(f"label_y: {label_y}")
            print(f"diff: {diff}")
            print(f"correct: {correct}")
    accuracy = correct / total
    print(f"accuracy: {accuracy}")
    return loss, accuracy


class ContactClient(fl.client.NumPyClient):

    # def __init__(self, cid, net):
    #     self.cid = cid
    #     self.net = net

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_loader, epochs=1)
        return self.get_parameters(config={}), len(train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, test_loader)
        return float(loss), len(test_loader), {"accuracy": float(accuracy)}


# def client_fn(cid) -> ContactClient:
#     # net = Net().to(DEVICE)
#     # trainloader = trainloaders[int(cid)]
#     # valloader = valloaders[int(cid)]
#     return ContactClient(cid)

fl.client.start_numpy_client(server_address="localhost:8082", client=ContactClient())

# if __name__ == '__main__':
#     fl.client.start_numpy_client(server_address="[::]:8082", client=ContactClient())
