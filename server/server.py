import flwr as fl
import torch
import torch.nn.functional as F
from torch import nn

INPUT_DIM = 2
OUTPUT_DIM = 3
HIDDEN_DIM = 6

NUM_CLIENTS = 3


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
        lstm_out, _ = self.lstm(torch.tensor(input, dtype=torch.float32))
        print(f"lstm_out{lstm_out[-1]}")
        tag_space = self.hidden2tag(lstm_out[-1])
        print(f"tag_space{tag_space}")
        tag_scores = F.softmax(tag_space, dim=0)
        print(f"tag_scores{tag_scores}")
        return tag_scores


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


Net = LSTMTagger(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# Create an instance of the model and get the parameters
params = get_parameters(Net)


# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(params),
)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}


# def client_fn(cid) -> ContactClient:
#     net = Net().to(DEVICE)
#     # trainloader = trainloaders[int(cid)]
#     # valloader = valloaders[int(cid)]
#     return ContactClient(cid, net)


# Start simulation
# fl.simulation.start_simulation(
#     client_fn=None,
#     num_clients=NUM_CLIENTS,
#     config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
#     strategy=strategy,
#     client_resources=client_resources,
# )

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8082",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)