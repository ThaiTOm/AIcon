# train_vit_head_only_ddp.py

# ==============================================================================
# PHẦN IMPORT THƯ VIỆN
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
import shutil
import random

# ### THAY ĐỔI CHO DDP ###
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# ### KẾT THÚC THAY ĐỔI ###


# ==============================================================================
# PHẦN CẤU HÌNH HUẤN LUYỆN
# ==============================================================================
SOURCE_DATA_DIR = "datasets"
PREPARED_DATA_DIR = os.path.join(SOURCE_DATA_DIR, "prepared_data")

# Tên tệp để lưu trọng số đã huấn luyện (thay đổi để không ghi đè)
WEIGHTS_SAVE_PATH = "bike_motorbike_vit_weights.pth"

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64  # Đây là batch size cho MỖI GPU
NUM_EPOCHS = 15
SPLIT_RATIO = 0.9

# ==============================================================================
# CÁC HÀM TIỆN ÍCH DDP VÀ DỮ LIỆU (KHÔNG THAY ĐỔI)
# ==============================================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def clean_image_dataset(root_dir):
    if not os.path.isdir(root_dir):
        print(f"Lỗi: Thư mục '{root_dir}' không tồn tại.")
        return
    print(f"Bắt đầu quá trình quét và dọn dẹp thư mục: {root_dir}")
    deleted_files_count = 0
    all_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            all_files.append(os.path.join(subdir, filename))
    print(f"Tìm thấy tổng cộng {len(all_files)} tệp để kiểm tra.")
    for file_path in tqdm(all_files, desc="Đang kiểm tra ảnh"):
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            continue
        try:
            with Image.open(file_path) as img:
                img.convert('RGB')
        except (IOError, OSError, UnidentifiedImageError) as e:
            print(f"\nPhát hiện tệp ảnh bị lỗi: {file_path} | Lỗi: {e}")
            try:
                os.remove(file_path)
                print(f"Đã xóa thành công tệp: {file_path}")
                deleted_files_count += 1
            except OSError as remove_error:
                print(f"Lỗi khi xóa tệp: {remove_error}")
    print(f"\nQUÁ TRÌNH DỌN DẸP CHO '{root_dir}' HOÀN TẤT! Đã xóa {deleted_files_count} tệp lỗi.")

def split_data_from_source(source_dir, prepared_dir, split_ratio):
    classes = ['bike', 'motorbike']
    if os.path.exists(os.path.join(prepared_dir, 'train')) and os.path.exists(os.path.join(prepared_dir, 'val')):
        print(f"Thư mục '{prepared_dir}' đã có cấu trúc train/val. Bỏ qua bước chia tách.")
        return
    print(f"Đang chuẩn bị và chia tách dữ liệu vào '{prepared_dir}'...")
    if os.path.exists(prepared_dir):
        shutil.rmtree(prepared_dir)
    for split in ['train', 'val']:
        for cls in classes:
            os.makedirs(os.path.join(prepared_dir, split, cls), exist_ok=True)
    for cls in classes:
        source_class_dir = os.path.join(source_dir, cls)
        if not os.path.isdir(source_class_dir):
            print(f"Cảnh báo: Không tìm thấy '{source_class_dir}'. Bỏ qua.")
            continue
        all_files = [f for f in os.listdir(source_class_dir) if os.path.isfile(os.path.join(source_class_dir, f))]
        random.shuffle(all_files)
        split_point = int(len(all_files) * split_ratio)
        train_files = all_files[:split_point]
        val_files = all_files[split_point:]
        for f in tqdm(train_files, desc=f"Sao chép {cls} vào train"):
            shutil.copy(os.path.join(source_class_dir, f), os.path.join(prepared_dir, 'train', cls, f))
        for f in tqdm(val_files, desc=f"Sao chép {cls} vào val"):
            shutil.copy(os.path.join(source_class_dir, f), os.path.join(prepared_dir, 'val', cls, f))
    print("Chia tách dữ liệu hoàn tất.")

def create_dataloaders(prepared_dataset_dir, rank):
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
    if rank == 0:
        # clean_image_dataset(os.path.join(prepared_dataset_dir, 'train', 'bike'))
        # clean_image_dataset(os.path.join(prepared_dataset_dir, 'train', 'motorbike'))
        # clean_image_dataset(os.path.join(prepared_dataset_dir, 'val', 'bike'))
        # clean_image_dataset(os.path.join(prepared_dataset_dir, 'val', 'motorbike'))
        pass
    dist.barrier()
    image_datasets = {x: datasets.ImageFolder(os.path.join(prepared_dataset_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    samplers = {
        'train': DistributedSampler(image_datasets['train'], shuffle=True),
        'val': DistributedSampler(image_datasets['val'], shuffle=False)
    }
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                 pin_memory=True, sampler=samplers[x])
                   for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    if rank == 0:
        print("Các lớp được tìm thấy:", class_names)
    if not (set(class_names) == {'bike', 'motorbike'}):
        raise ValueError("Các lớp trong bộ dữ liệu phải là 'bike' và 'motorbike'.")
    return dataloaders, class_names, samplers['train']


# ==============================================================================
# PHẦN HUẤN LUYỆN (KHÔNG THAY ĐỔI)
# ==============================================================================
def train_model(model, criterion, optimizer, dataloaders, train_sampler, num_epochs, local_rank, rank):
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            iterable = dataloaders[phase]
            if rank == 0:
                iterable = tqdm(iterable, desc=f"{phase.capitalize()} Phase")
            for inputs, labels in iterable:
                inputs = inputs.to(local_rank)
                labels = labels.to(local_rank)
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
            total_loss = torch.tensor(running_loss).to(local_rank)
            total_corrects = torch.tensor(running_corrects).to(local_rank)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_corrects, op=dist.ReduceOp.SUM)
            epoch_loss = total_loss / len(dataloaders[phase].dataset)
            epoch_acc = total_corrects.double() / len(dataloaders[phase].dataset)
            if rank == 0:
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.module.state_dict(), WEIGHTS_SAVE_PATH)
                    print(f"Độ chính xác tốt nhất mới: {best_acc:.4f}. Đã lưu trọng số vào '{WEIGHTS_SAVE_PATH}'!")
    return model


# ==============================================================================
# HÀM MAIN
# ==============================================================================
if __name__ == '__main__':
    setup_ddp()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Bắt đầu process rank {rank} trên GPU {local_rank}.")

    if rank == 0:
        split_data_from_source(SOURCE_DATA_DIR, PREPARED_DATA_DIR, SPLIT_RATIO)
    dist.barrier()

    dataloaders, class_names, train_sampler = create_dataloaders(PREPARED_DATA_DIR, rank)

    # 3. Tải mô hình ViT-B-16
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    # ### <<< THAY ĐỔI QUAN TRỌNG: ĐÓNG BĂNG CÁC LỚP GỐC ###
    if rank == 0:
        print("Đóng băng tất cả các tham số của mô hình gốc...")
    for param in model.parameters():
        param.requires_grad = False
    # ### KẾT THÚC THAY ĐỔI ###

    # 4. Thay thế lớp phân loại cuối cùng.
    # Các tham số của lớp mới này sẽ có `requires_grad=True` theo mặc định.
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, len(class_names))
    if rank == 0:
        print("Đã thay thế lớp head. Chỉ có các tham số của lớp head mới sẽ được huấn luyện.")

    # 5. Định nghĩa hàm mất mát và bộ tối ưu hóa
    criterion = nn.CrossEntropyLoss()

    # ### <<< THAY ĐỔI QUAN TRỌNG: CHỈ TỐI ƯU HÓA CÁC THAM SỐ CỦA HEAD ###
    # Thay vì `model.parameters()`, chúng ta chỉ truyền các tham số của lớp cuối cùng.
    # Điều này hiệu quả hơn và đảm bảo chỉ có head được cập nhật.
    optimizer = optim.Adam(model.heads.head.parameters(), lr=LEARNING_RATE)
    if rank == 0:
        print("Optimizer được cấu hình để chỉ cập nhật các tham số của lớp head.")
    # ### KẾT THÚC THAY ĐỔI ###

    # Chuyển mô hình đến GPU tương ứng
    model = model.to(local_rank)

    # Bọc mô hình với DDP
    model = DDP(model, device_ids=[local_rank])
    if rank == 0:
        print("Mô hình đã được bọc bằng DDP.")

    # Bắt đầu huấn luyện
    if rank == 0:
        print("\nBắt đầu quá trình huấn luyện (chỉ head)...")
    train_model(model, criterion, optimizer, dataloaders, train_sampler,
                num_epochs=NUM_EPOCHS, local_rank=local_rank, rank=rank)

    if rank == 0:
        print("\nHuấn luyện hoàn tất.")

    cleanup_ddp()