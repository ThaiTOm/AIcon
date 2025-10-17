# train_vit.py

# ==============================================================================
# PHẦN IMPORT THƯ VIỆN
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
import shutil
import random

# ==============================================================================
# PHẦN CẤU HÌNH HUẤN LUYỆN
# ==============================================================================
# Thư mục chứa các thư mục lớp gốc (ví dụ: 'datasets/bike', 'datasets/motorbike')
SOURCE_DATA_DIR = "datasets"
# Thư mục đích để chứa bộ dữ liệu đã được chia train/val
PREPARED_DATA_DIR = os.path.join(SOURCE_DATA_DIR, "prepared_data")

# Tên tệp để lưu trọng số đã huấn luyện
WEIGHTS_SAVE_PATH = "bike_motorbike_vit_weights.pth"

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 15
SPLIT_RATIO = 0.8  # 80% cho huấn luyện, 20% cho xác thực

# Thiết bị (sử dụng GPU nếu có)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Sử dụng thiết bị: {DEVICE}")


# ==============================================================================
# PHẦN CHUẨN BỊ VÀ CHIA TÁCH DỮ LIỆU
# ==============================================================================
def split_data_from_source(source_dir, prepared_dir, split_ratio):
    """
    Tự động chia dữ liệu từ các thư mục lớp gốc thành cấu trúc train/val.
    """
    classes = ['bike', 'motorbike']

    # Kiểm tra xem dữ liệu đã được chuẩn bị chưa để tránh làm lại
    if os.path.exists(os.path.join(prepared_dir, 'train')) and os.path.exists(os.path.join(prepared_dir, 'val')):
        print(f"Thư mục '{prepared_dir}' đã có cấu trúc train/val. Bỏ qua bước chia tách.")
        return

    print(f"Đang chuẩn bị và chia tách dữ liệu vào '{prepared_dir}'...")

    # Xóa thư mục cũ nếu có để đảm bảo sự sạch sẽ
    if os.path.exists(prepared_dir):
        shutil.rmtree(prepared_dir)

    # Tạo các thư mục train/val cần thiết
    for split in ['train', 'val']:
        for cls in classes:
            os.makedirs(os.path.join(prepared_dir, split, cls), exist_ok=True)

    for cls in classes:
        source_class_dir = os.path.join(source_dir, cls)

        # Kiểm tra xem thư mục lớp nguồn có tồn tại không
        if not os.path.isdir(source_class_dir):
            print(f"Cảnh báo: Không tìm thấy thư mục nguồn '{source_class_dir}'. Bỏ qua lớp này.")
            continue

        all_files = [f for f in os.listdir(source_class_dir) if os.path.isfile(os.path.join(source_class_dir, f))]
        random.shuffle(all_files)

        split_point = int(len(all_files) * split_ratio)
        train_files = all_files[:split_point]
        val_files = all_files[split_point:]

        # Sao chép tệp vào thư mục train
        for f in tqdm(train_files, desc=f"Sao chép {cls} vào train"):
            shutil.copy(os.path.join(source_class_dir, f), os.path.join(prepared_dir, 'train', cls, f))

        # Sao chép tệp vào thư mục val
        for f in tqdm(val_files, desc=f"Sao chép {cls} vào val"):
            shutil.copy(os.path.join(source_class_dir, f), os.path.join(prepared_dir, 'val', cls, f))

    print("Chia tách dữ liệu hoàn tất.")


def create_dataloaders(prepared_dataset_dir):
    """Tạo dataloaders từ thư mục dữ liệu đã được chuẩn bị."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(prepared_dataset_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    print("Các lớp được tìm thấy:", class_names)
    if not (set(class_names) == {'bike', 'motorbike'}):
        raise ValueError("Các lớp trong bộ dữ liệu phải là 'bike' và 'motorbike'.")

    return dataloaders, class_names


# ==============================================================================
# PHẦN HUẤN LUYỆN (Không thay đổi)
# ==============================================================================
def train_model(model, criterion, optimizer, dataloaders, num_epochs=10):
    """Hàm chính để huấn luyện mô hình."""
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
                print(f"Độ chính xác xác thực tốt nhất mới: {best_acc:.4f}. Đã lưu trọng số vào '{WEIGHTS_SAVE_PATH}'!")

    return model


# ==============================================================================
# HÀM MAIN
# ==============================================================================
if __name__ == '__main__':
    # 1. Tự động chia tách dữ liệu từ thư mục nguồn
    split_data_from_source(SOURCE_DATA_DIR, PREPARED_DATA_DIR, SPLIT_RATIO)

    # 2. Tạo Dataloaders từ dữ liệu đã được chuẩn bị
    dataloaders, class_names = create_dataloaders(PREPARED_DATA_DIR)

    # 3. Tải mô hình ViT-B-16 đã được huấn luyện trước
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    # 4. Thay thế lớp phân loại cuối cùng
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, len(class_names))

    model = model.to(DEVICE)
    print("Mô hình đã được sửa đổi cho 2 lớp.")

    # 5. Định nghĩa hàm mất mát và bộ tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Bắt đầu huấn luyện
    print("\nBắt đầu quá trình huấn luyện...")
    train_model(model, criterion, optimizer, dataloaders, num_epochs=NUM_EPOCHS)

    print("\nHuấn luyện hoàn tất.")