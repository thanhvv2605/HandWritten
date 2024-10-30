import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import random
import csv
import os

# Kiểm tra thiết bị MPS (GPU của Apple Silicon)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Using device: {device}")

# Đọc dữ liệu từ file CSV
with open('hand_written.csv', 'r') as csv_file:
    result = csv.reader(csv_file)
    rows = []
    for row in result:
        rows.append(row)

# Lọc dữ liệu và lấy những nhãn cần thiết
train_data = []
train_label = []
for letter in rows:
    if (letter[0] == '0') or (letter[0] == '1'):
        x = np.array([int(j) for j in letter[1:]])
        x = x.reshape(28, 28)
        train_data.append(x)
        train_label.append(int(letter[0]))
    else:
        break

# Chuyển thành numpy arrays
train_data = np.array(train_data)
train_label = np.array(train_label)

# Shuffle dữ liệu
shuffle_order = list(range(len(train_label)))
random.shuffle(shuffle_order)
train_data = train_data[shuffle_order]
train_label = train_label[shuffle_order]

# Chia dữ liệu thành tập huấn luyện, kiểm thử và xác thực
train_x, test_x, train_y, test_y = train_test_split(train_data, train_label, test_size=0.4, random_state=42)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.4, random_state=42)

# Chuyển reshape dữ liệu thành dạng NCHW (batch_size, channels, height, width)
train_x = train_x.reshape(-1, 1, 28, 28)
val_x = val_x.reshape(-1, 1, 28, 28)
test_x = test_x.reshape(-1, 1, 28, 28)

# Chuyển đổi numpy arrays thành PyTorch tensors và chuyển sang thiết bị MPS nếu có
train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
val_x = torch.tensor(val_x, dtype=torch.float32).to(device)
test_x = torch.tensor(test_x, dtype=torch.float32).to(device)

train_y = torch.tensor(train_y, dtype=torch.long).to(device)
val_y = torch.tensor(val_y, dtype=torch.long).to(device)
test_y = torch.tensor(test_y, dtype=torch.long).to(device)

# Định nghĩa mô hình CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Khởi tạo mô hình và chuyển mô hình sang MPS hoặc CPU
model = ConvNet().to(device)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Tạo DataLoader
BATCH_SIZE = 64
train_data = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data = torch.utils.data.TensorDataset(val_x, val_y)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# Vòng lặp huấn luyện
N_EPOCHS = 50
best_val_loss = float('inf')

for epoch in range(N_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            loss = criterion(val_outputs, val_labels)
            val_loss += loss.item()
            _, val_predicted = torch.max(val_outputs, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Early stopping và lưu mô hình tốt nhất
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_mps.pth')

# Kiểm tra trên tập test
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    outputs = model(test_x)
    _, predicted = torch.max(outputs, 1)
    test_total = test_y.size(0)
    test_correct = (predicted == test_y).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")