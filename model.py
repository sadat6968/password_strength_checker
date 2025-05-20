import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Define PyTorch Model
class PasswordStrengthModel(nn.Module):
    def __init__(self, input_size):
        super(PasswordStrengthModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Convert password to feature vector
def encode_password(password):
    return np.array([
        len(password),
        sum(c.isdigit() for c in password),
        sum(c.isupper() for c in password),
        sum(not c.isalnum() for c in password)
    ], dtype=np.float32)

# Load dataset
df = pd.read_csv("password_dataset.csv")
X_train = np.array([encode_password(p) for p in df["password"]])
y_train = np.array(df["strength"], dtype=np.float32).reshape(-1, 1)

# Convert to tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

# Initialize model
input_size = 4
model = PasswordStrengthModel(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Train model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Function to evaluate password strength
def evaluate_password(password):
    model.eval()
    password_features = torch.tensor(encode_password(password))
    with torch.no_grad():
        prediction = model(password_features)
        strength = "Strong" if prediction.item() > 0.5 else "Weak"

    suggestions = []
    if len(password) < 8: suggestions.append("Make the password longer.")
    if not any(c.isupper() for c in password): suggestions.append("Add uppercase letters.")
    if not any(c.isdigit() for c in password): suggestions.append("Include numbers.")
    if not any(not c.isalnum() for c in password): suggestions.append("Use special characters.")

    return strength, suggestions
