import torch
import torch.nn as nn
import torch.optim as optim

class DLTrainer:
    def __init__(self, model, lr=0.001, epochs=50):
        self.model = model
        self.epochs = epochs
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            preds = self.model(X)
            loss = self.criterion(preds, y)

            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X).numpy()
