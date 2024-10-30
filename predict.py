import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch
import cv2  # OpenCV để đọc hình ảnh
import numpy as np
import matplotlib.pyplot as plt
import os

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

# Bước 1: Đọc hình ảnh và xử lý nó
def load_and_preprocess_image(image_path):
    # Đọc hình ảnh từ file (sử dụng OpenCV)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dưới dạng grayscale
    
    # Resize hình ảnh về kích thước 28x28
    img = cv2.resize(img, (28, 28))
    
    # Hiển thị hình ảnh để kiểm tra
    plt.imshow(img, cmap='gray')
    plt.title('Input Image')
    plt.show()
    
    # Chuẩn hóa pixel: biến đổi giá trị pixel từ [0, 255] về [0, 1]
    img = img / 255.0
    
    # Đổi thành tensor: thêm batch dimension và channel dimension
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    
    return img

# Bước 2: Dự đoán ký tự
def predict_character(model, image_tensor):
    # Đặt mô hình vào chế độ đánh giá
    model.eval()
    
    # Tắt gradient vì chỉ cần dự đoán
    with torch.no_grad():
        # Thực hiện dự đoán
        logits = model(image_tensor)
                
        # Lấy nhãn dự đoán có xác suất cao nhất
        _, predicted_label = torch.max(logits, 1)
    
    # Chuyển đổi sang numpy để in ra nhãn dự đoán
    predicted_label = predicted_label.item()  # Chuyển từ tensor về giá trị số
    
    return predicted_label

# Bước 3: Sử dụng mô hình đã lưu để dự đoán ký tự trong một hình ảnh
def main():
    # Tải mô hình đã huấn luyện
    model = ConvNet()  # Khởi tạo lại kiến trúc mô hình
    model.load_state_dict(torch.load('best_model_mps.pth', weights_only=True))  # Tải trọng số đã huấn luyện
    image_path = 'testB.jpg' 
    image_tensor = load_and_preprocess_image(image_path)
    
    # Dự đoán ký tự trong hình ảnh
    predicted_character = predict_character(model, image_tensor)
    
    # In ra kết quả dự đoán
    if predicted_character == 0:
        print(f"Ký tự được dự đoán là A")
    elif predicted_character == 1:
        print(f"Ký tự được dự đoán là B")
    elif predicted_character == 2:
        print(f"Ký tự được dự đoán là C")

# Gọi hàm chính
if __name__ == "__main__":
    main()