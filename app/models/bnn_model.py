import torch
import torch.nn as nn
import torchbnn as bnn

class BayesianYieldNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        prior_mu = 0.0
        prior_sigma = 0.1
        
        self.model = nn.Sequential(
            bnn.BayesLinear(prior_mu, prior_sigma, input_dim, 128),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu, prior_sigma, 128, 64),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu, prior_sigma, 64, 1)
        )

    def forward(self, x):
        return self.model(x)

def predict(X_tensor, device):
    model = BayesianYieldNN(input_dim=X_tensor.shape[1]).to(device)
    model.load_state_dict(torch.load("models/best_bnn_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        return model(X_tensor).cpu().numpy()