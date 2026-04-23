import torch
import torch.nn as nn
import torch.optim as optim

from data_generator import big_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaterialClassifier(nn.Module):

    def __init__(self, input_values, hidden_values, output_values):
        super(MaterialClassifier, self).__init__()

        # Layer 1 (Input to Hidden)
        self.hidden = nn.Linear(input_values,  hidden_values)

        # Layer 2 (Hidden to Output)
        self.output = nn.Linear(hidden_values, output_values)

        # Activation Function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))

        return x
    
raw_inputs = [
    [0.991, 0.013, 0.50], # Batch 1 (Pass)
    [0.921, 0.366, 0.48], # Batch 2 (Fail)
    [0.990, 0.006, 0.52], # Batch 3 (Pass)
    [0.936, 0.033, 0.50], # Batch 4 (Fail - Imputed 93.6)
    [0.850, 0.800, 0.30]  # Batch 5 (Fail)
]

labels = [[1], [0], [1], [0], [0]]

# X = torch.tensor(raw_inputs, dtype=torch.float32)
# y = torch.tensor(labels, dtype=torch.float32)

X, y = big_data
X, y = X.to(device), y.to(device)

model = MaterialClassifier(3, 8, 1).to(device)

# Loss function and Optimizer
bce_loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
print(device)
for epoch in range(1500):
    predictions = model(X)
    loss = bce_loss(predictions, y)
    optimizer.zero_grad() # Clear previous gradients
    loss.backward() # Calculate new gradients
    optimizer.step() # Tune the weights


    binary_preds = (predictions > 0.5).float()
    accuracy = (binary_preds == y).float().mean()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Loss {loss.item():.4f} | Accuracy {accuracy:.2%}")


test_batch = torch.tensor([[0.971, 0.099, 0.50]], dtype=torch.float32).to(device)
borderline = torch.tensor([[0.915, 0.32, 0.50]], dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    prediction = model(test_batch)
    print(f"\nPassed Batch Prediction: {prediction.item():.2%} chance passing.")
    prediction = model(borderline)
    print(f"\nFailed Batch Prediction: {prediction.item():.2%} chance passing.")