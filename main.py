# import numpy as np
# import csv
# import matplotlib.pyplot as plt
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
# import torch
# import cv2  # OpenCV để đọc hình ảnh
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
# # Test DEMO 1/11/2024 Lần 1


print('Vo Van Thanh 2/11')

# # # Đọc dữ liệu từ file CSV
# # with open('hand_written.csv', 'r') as csv_file:
# #     result = csv.reader(csv_file)
# #     rows = []
# #     for row in result:
# #         rows.append(row)
# #
# # train_data = []
# # train_label = []
# # # for letter in rows:
# # #         x = np.array([int(j) for j in letter[1:]])
# # #         x = x.reshape(28, 28)
# # #         train_data.append(x)
# # #         train_label.append(int(letter[0]))
# #
# # train_data = [] # dữ liệu training
# # train_label = [] # label của chúng
# #
# # for letter in rows:
# #     if (letter[0] == '0') or (letter[0] == '1') or (letter[0] == '2') or (letter[0] == '3'):
# #         x = np.array([int(j) for j in letter[1:]])
# #         x = x.reshape(28, 28)
# #         train_data.append(x)
# #         train_label.append(int(letter[0]))
# #     else:
# #         break
# #
# #
# # # Chuyển thành numpy arrays
# # train_data = np.array(train_data)
# # train_label = np.array(train_label)
# #
# # # Shuffle dữ liệu
# # shuffle_order = list(range(len(train_label)))
# # random.shuffle(shuffle_order)
# # train_data = train_data[shuffle_order]
# # train_label = train_label[shuffle_order]
# #
# # # Chia dữ liệu thành tập huấn luyện, kiểm thử và xác thực
# # train_x, test_x, train_y, test_y = train_test_split(train_data, train_label, test_size=0.2, random_state=42)
# # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

# # Đường dẫn tới dataset
# dataset_path = '/kaggle/input/az-handwritten-alphabets-in-csv-format'

# # Kiểm tra các file có trong dataset để xác định tên file CSV
# print("Files in dataset directory:", os.listdir(dataset_path))

# # Đọc file CSV (giả sử tên file là 'A_Z Handwritten Data.csv', kiểm tra tên chính xác từ bước trên)
# csv_file_path = os.path.join(dataset_path, 'A_Z Handwritten Data.csv')
# data = pd.read_csv(csv_file_path)

# # Hiển thị thông tin cơ bản về dataset
# print("Dataset loaded successfully!")
# print("Dataset shape:", data.shape)
# print("First few rows of the dataset:")
# print(data.head())

# # Tách dữ liệu và nhãn từ dataset
# # Giả sử cột đầu tiên là nhãn và các cột còn lại là dữ liệu pixel
# rows = data.values.tolist()  # Chuyển DataFrame thành list để lặp qua từng dòng
# train_data = []  # dữ liệu training
# train_label = []  # label của chúng

# # Duyệt qua các hàng trong dữ liệu để lấy nhãn và dữ liệu
# for letter in rows:
#     if letter[0] in [0, 1, 2, 3]:  # lấy các nhãn từ 0-3
#         x = np.array([int(j) for j in letter[1:]])  # lấy các cột dữ liệu (pixel)
#         x = x.reshape(28, 28)  # reshape thành 28x28 (giả sử ảnh có kích thước 28x28)
#         train_data.append(x)
#         train_label.append(int(letter[0]))
#     else:
#         break  # dừng nếu nhãn không phải là 0, 1, 2, hoặc 3

# # Chuyển thành numpy arrays
# train_data = np.array(train_data)
# train_label = np.array(train_label)

# # Shuffle dữ liệu
# shuffle_order = list(range(len(train_label)))
# random.shuffle(shuffle_order)
# train_data = train_data[shuffle_order]
# train_label = train_label[shuffle_order]

# # Chia dữ liệu thành tập huấn luyện, kiểm thử và xác thực
# train_x, test_x, train_y, test_y = train_test_split(train_data, train_label, test_size=0.2, random_state=42)
# train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

# # Kiểm tra kích thước của các tập dữ liệu
# print("Training set size:", train_x.shape)
# print("Validation set size:", val_x.shape)
# print("Test set size:", test_x.shape)
# # Chuyển reshape dữ liệu thành dạng NCHW (batch_size, channels, height, width)
# train_x = train_x.reshape(-1, 1, 28, 28)
# val_x = val_x.reshape(-1, 1, 28, 28)
# test_x = test_x.reshape(-1, 1, 28, 28)

# # Chuyển đổi numpy arrays thành PyTorch tensors
# train_x = torch.tensor(train_x, dtype=torch.float32)
# val_x = torch.tensor(val_x, dtype=torch.float32)
# test_x = torch.tensor(test_x, dtype=torch.float32)

# train_y = torch.tensor(train_y, dtype=torch.long)
# val_y = torch.tensor(val_y, dtype=torch.long)
# test_y = torch.tensor(test_y, dtype=torch.long)

# original_test_y = test_y
# # Định nghĩa mô hình CNN
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 7 * 7, 1024)
#         self.fc2 = nn.Linear(1024, 26)
#         self.dropout = nn.Dropout(0.5)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 7 * 7)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# # Khởi tạo mô hình
# model = ConvNet()

# # Loss function và optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Tạo DataLoader
# BATCH_SIZE = 32
# train_data = torch.utils.data.TensorDataset(train_x, train_y)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# val_data = torch.utils.data.TensorDataset(val_x, val_y)
# val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# # Vòng lặp huấn luyện
# N_EPOCHS = 5
# # N_EPOCHS = 50
# for epoch in range(N_EPOCHS):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     train_loss = running_loss / len(train_loader)
#     train_accuracy = 100 * correct / total
#     print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

#     # Validation
#     model.eval()
#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0
    
#     with torch.no_grad():
#         for val_inputs, val_labels in val_loader:
#             val_outputs = model(val_inputs)
#             loss = criterion(val_outputs, val_labels)
#             val_loss += loss.item()
#             _, val_predicted = torch.max(val_outputs, 1)
#             val_total += val_labels.size(0)
#             val_correct += (val_predicted == val_labels).sum().item()

#     val_loss = val_loss / len(val_loader)
#     val_accuracy = 100 * val_correct / val_total
#     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


# # Lưu các tham số của mô hình (state_dict)
# torch.save(model.state_dict(), 'mymodel.pth')
# # Khởi tạo lại kiến trúc mô hình
# model = ConvNet()  

# # Tải tham số vào mô hình
# model.load_state_dict(torch.load('mymodel.pth', weights_only=True))


# # Chuyển mô hình sang chế độ đánh giá (không cần huấn luyện nữa)
# model.eval()

# # Chuyển đổi test_x thành tensor nếu chưa
# test_x = torch.tensor(test_x, dtype=torch.float32)

# # Dự đoán với tập dữ liệu test
# with torch.no_grad():  # Tắt tính toán gradient vì chỉ cần dự đoán
#     test_logits = model(test_x)

# # Lấy phần tử có giá trị lớn nhất
# _, test_logits = torch.max(test_logits, 1)

# # Chuyển test_logits về numpy array để tính toán kết quả
# test_logits = test_logits.cpu().numpy()

# # So sánh với nhãn gốc (original_test_y)
# accuracy = np.sum(test_logits == test_y.numpy()) / len(test_logits)
# print(f"Accuracy: {accuracy}")



# # dự đoán

# # Bước 1: Đọc hình ảnh và xử lý nó
# image_path = 'input_image.png'
# # image_path = '/Users/apple/Documents/E7 Document HUST/BKAI_Lab/Deep Learning/HandWritten/input_image.jpg'
# # if not os.path.exists(image_path):
# #         print(f"Error: File {image_path} not found.")
        
# def load_and_preprocess_image(image_path):
#     # Đọc hình ảnh từ file (sử dụng OpenCV)
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dưới dạng grayscale
    
#     print(img)
#     # Resize hình ảnh về kích thước 28x28
#     img = cv2.resize(img, (28, 28))
    
#     # Hiển thị hình ảnh để kiểm tra
#     plt.imshow(img, cmap='gray')
#     plt.title('Input Image')
#     plt.show()
    
#     # Chuẩn hóa pixel: biến đổi giá trị pixel từ [0, 255] về [0, 1]
#     img = img / 255.0
    
#     # Đổi thành tensor: thêm batch dimension và channel dimension
#     img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    
#     return img

# # Bước 2: Dự đoán ký tự
# def predict_character(model, image_tensor):
#     # Đặt mô hình vào chế độ đánh giá
#     model.eval()
    
#     # Tắt gradient vì chỉ cần dự đoán
#     with torch.no_grad():
#         # Thực hiện dự đoán
#         logits = model(image_tensor)
        
#         # Lấy nhãn dự đoán có xác suất cao nhất
#         _, predicted_label = torch.max(logits, 1)
    
#     # Chuyển đổi sang numpy để in ra nhãn dự đoán
#     predicted_label = predicted_label.item()  # Chuyển từ tensor về giá trị số
    
#     return predicted_label

# # Bước 3: Sử dụng mô hình đã lưu để dự đoán ký tự trong một hình ảnh
# def main():
#     # Tải mô hình đã huấn luyện
#     model = ConvNet()  # Khởi tạo lại kiến trúc mô hình
#     model.load_state_dict(torch.load('mymodel.pth', weights_only=True))  # Tải trọng số đã huấn luyện
    
#     # Đọc và xử lý hình ảnh đầu vào
#     image_path = 'input_image.png'  # Thay bằng đường dẫn đến hình ảnh của bạn
#     image_tensor = load_and_preprocess_image(image_path)
    
#     # Dự đoán ký tự trong hình ảnh
#     predicted_character = predict_character(model, image_tensor)
    
#     # In ra kết quả dự đoán
#     print(f"Ký tự được dự đoán: {predicted_character}")
#     p

# # Gọi hàm chính
# if __name__ == "__main__":
#     main()