import torch
import torch.nn as nn

class SimpleLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)   
        lstm_out, (h_n, _) = self.lstm(x)
        final_hidden = h_n[-1]
        return self.fc(final_hidden)

def predict(X_tensor, device):
    model = SimpleLSTMModel(input_dim=X_tensor.shape[1]).to(device)
    model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        return model(X_tensor).cpu().numpy()