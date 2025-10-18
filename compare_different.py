# ==============================================================================
# PHẦN IMPORT THƯ VIỆN
# ==============================================================================
import os
import cv2
import json
import glob
import time
from collections import defaultdict, Counter
from ultralytics import YOLO
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==============================================================================
# PHẦN CẤU HÌNH VÀ KHỞI TẠO MODEL
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. CẤU HÌNH MODEL ViT CHUNG (PRE-TRAINED TRÊN IMAGENET)
# ------------------------------------------------------------------------------
# Định nghĩa ánh xạ từ ImageNet sang các lớp tùy chỉnh
CUSTOM_CLASSES = {
    'car': [
        'sports car, sport car', 'convertible', 'jeep, landrover', 'limousine, limo',
        'minivan', 'racer, race car, racing car', 'cab', 'hack', 'taxi', 'taxicab',
        'ambulance', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
        'recreational vehicle, RV, R.V.', 'station wagon, wagon, estate car, beach wagon, station waggon, waggon',
        'passenger car, coach, carriage', 'car', 'truck'
    ],
    'motorcycle': ['motor scooter, scooter', 'moped', 'motorcycle'],
    'bicycle': ['mountain bike, all-terrain bike, off-roader', 'bicycle-built-for-two, tandem bicycle, tandem',
                'unicycle, monocycle', "bicycle"],
    'person': ['scuba diver', 'groom, bridegroom', 'baseball player, ballplayer', 'skier', "person"]
}
# Tạo map tra cứu ngược
imagenet_to_custom_map = {label: custom_class for custom_class, labels in CUSTOM_CLASSES.items() for label in labels}

# Tải mô hình ViT ImageNet (chỉ một lần)
print("Đang tải mô hình Vision Transformer (ViT) từ ImageNet...")
vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
vit_model = models.vit_b_16(weights=vit_weights)
vit_model.eval()
vit_preprocess = vit_weights.transforms()
imagenet_categories = vit_weights.meta["categories"]
print("Mô hình ViT ImageNet đã được tải.")

# ------------------------------------------------------------------------------
# 2. CẤU HÌNH MODEL ViT TÙY CHỈNH (MODEL BẠN ĐÃ HUẤN LUYỆN)
# ------------------------------------------------------------------------------
CUSTOM_VIT_WEIGHTS_PATH = "bike_motorbike_vit_weights.pth"
# Các lớp mà model tùy chỉnh của bạn đã được huấn luyện, THEO ĐÚNG THỨ TỰ.
# Dựa trên script huấn luyện của bạn, thứ tự là ['bike', 'motorbike']
CUSTOM_VIT_CLASSES = ['bike', 'motorbike']
# Ánh xạ từ output của model tùy chỉnh sang nhãn chuẩn hóa
CUSTOM_VIT_LABEL_MAP = {
    'bike': 'bicycle',
    'motorbike': 'motorcycle'
}


def load_custom_vit_model(weights_path, num_classes):
    """Hàm để tải kiến trúc ViT và áp trọng số đã huấn luyện của bạn."""
    if not os.path.exists(weights_path):
        print(f"LỖI: Không tìm thấy tệp trọng số '{weights_path}'. Vui lòng đảm bảo tệp này tồn tại.")
        return None, None

    print(f"Đang tải mô hình ViT tùy chỉnh từ '{weights_path}'...")
    model = models.vit_b_16(weights=None)  # Khởi tạo không có trọng số pre-trained

    # Thay thế lớp phân loại cuối cùng để khớp với số lớp của bạn (là 2)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)

    # Tải trọng số đã huấn luyện
    # Dùng map_location để đảm bảo model tải được trên cả CPU và GPU
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cuda:0')))
    model.eval()

    # Tạo bộ tiền xử lý ảnh (phải giống với bộ validation trong lúc train)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Mô hình ViT tùy chỉnh đã được tải thành công.")
    return model, preprocess


# Tải model ViT tùy chỉnh
custom_vit_model, custom_vit_preprocess = load_custom_vit_model(
    CUSTOM_VIT_WEIGHTS_PATH,
    len(CUSTOM_VIT_CLASSES)
)


# ------------------------------------------------------------------------------
# 3. CÁC HÀM PHÂN LOẠI
# ------------------------------------------------------------------------------
### <<< THAY ĐỔI: Hàm trả về (nhãn, xác suất) thay vì chỉ nhãn
def classify_with_vit(image_crop_np, device='cuda:0'):
    """
    Phân loại ảnh bằng model ViT pre-trained trên ImageNet.
    Trả về: (string, float) -> (nhãn tùy chỉnh, xác suất)
    """
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image_crop_np, cv2.COLOR_BGR2RGB))
        img_tensor = vit_preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = vit_model(img_tensor)
            # Chuyển đổi logits sang xác suất bằng softmax
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # Lấy xác suất cao nhất và chỉ số của nó
            max_prob, prediction_index_tensor = torch.max(probabilities, 1)
            prediction_index = prediction_index_tensor.item()
            confidence = max_prob.item()

        predicted_imagenet_label = imagenet_categories[prediction_index]
        custom_label = imagenet_to_custom_map.get(predicted_imagenet_label, "other")
        return custom_label, confidence
    except Exception:
        return "other", 0.0

### <<< THAY ĐỔI: Hàm trả về (nhãn, xác suất) thay vì chỉ nhãn
def classify_bike_motorbike_with_custom_vit(image_crop_np, device='cuda:0'):
    """
    Phân loại ảnh bằng model ViT tùy chỉnh để phân biệt bike/motorbike.
    Trả về: (string, float) -> (nhãn tùy chỉnh, xác suất)
    """
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image_crop_np, cv2.COLOR_BGR2RGB))
        img_tensor = custom_vit_preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = custom_vit_model(img_tensor)
            # Chuyển đổi logits sang xác suất bằng softmax
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # Lấy xác suất cao nhất và chỉ số của nó
            max_prob, prediction_index_tensor = torch.max(probabilities, 1)
            prediction_index = prediction_index_tensor.item()
            confidence = max_prob.item()

        predicted_label = CUSTOM_VIT_CLASSES[prediction_index]
        # Sử dụng map để trả về nhãn nhất quán ('bicycle' hoặc 'motorcycle')
        final_label = CUSTOM_VIT_LABEL_MAP.get(predicted_label, 'other')
        return final_label, confidence
    except Exception:
        return "other", 0.0


# ==============================================================================
# PHẦN CẤU HÌNH XỬ LÝ VIDEO
# ==============================================================================
INPUT_VIDEO_DIR = "input_videos"
OUTPUT_VIDEO_DIR = "output_videos"
OUTPUT_FRAMES_DIR = "output_frames"
OUTPUT_JSON_FILE = "results.json"
MODEL_NAME = 'yolo12x.pt'

# Xóa các thư mục output cũ
for dir_path in [OUTPUT_FRAMES_DIR, OUTPUT_VIDEO_DIR]:
    if os.path.exists(dir_path):
        import shutil

        shutil.rmtree(dir_path)


# ==============================================================================
# PHẦN XỬ LÝ CHÍNH
# ==============================================================================
def analyze_videos_single_device():
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")

    # Chuyển các model đến đúng thiết bị
    vit_model.to(device)
    if custom_vit_model:
        custom_vit_model.to(device)
    else:
        print("CẢNH BÁO: Không thể tải model ViT tùy chỉnh. Chức năng phân loại bike/motorbike sẽ không hoạt động.")
        return  # Dừng nếu không tải được model tùy chỉnh

    # 1. Tải mô hình YOLO
    print(f"Đang tải mô hình {MODEL_NAME}...")
    yolo_model = YOLO(MODEL_NAME)
    CLASS_NAMES = yolo_model.names
    print(CLASS_NAMES)

    print("Mô hình YOLO đã được tải thành công.")

    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

    video_paths = glob.glob(os.path.join(INPUT_VIDEO_DIR, '*'))
    if not video_paths:
        print(f"Không tìm thấy video nào trong thư mục '{INPUT_VIDEO_DIR}'.")
        return

    final_results = {str(i): {} for i in range(1, 9)}

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n==============================================")
        print(f"Bắt đầu xử lý video: {video_name}")
        print(f"==============================================")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Danh sách để lưu kết quả phát hiện của từng khung hình
        frame_by_frame_results = []

        # ==============================================================================
        # GIAI ĐOẠN 1: PHÁT HIỆN, PHÂN LOẠI VÀ THU THẬP DỮ LIỆU
        # ==============================================================================
        print("Giai đoạn 1: Phát hiện và thu thập dữ liệu...")
        for frame_idx in tqdm(range(total_frames), desc=f"Pass 1: Detecting {video_name}"):
            ret, frame = cap.read()
            if not ret: break

            # THAY ĐỔI: Sử dụng detect thay vì track
            yolo_results = yolo_model.predict(frame, conf=0.1, verbose=False)[0]

            current_frame_objects = []
            if yolo_results.boxes is not None:
                for box in yolo_results.boxes:
                    # Không còn tracker_id khi dùng detect
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_label_id = int(box.cls[0])
                    yolo_label = CLASS_NAMES.get(yolo_label_id, "unknown")
                    yolo_confidence = float(box.conf[0])

                    # =========================================================
                    # ========= LOGIC PHÂN LOẠI ĐÃ ĐƯỢC CẬP NHẬT =========
                    # =========================================================
                    final_label = "other"

                    # Ưu tiên 1: Nếu YOLO nhận diện là motor/bike, luôn dùng model ViT tùy chỉnh để quyết định
                    if yolo_label in ['motorcycle', 'bicycle']:
                        cropped_object = frame[y1:y2, x1:x2]
                        if cropped_object.size > 0:
                            final_label, _ = classify_bike_motorbike_with_custom_vit(cropped_object, device)
                    # Mặc định: Với các lớp còn lại, so sánh xác suất của YOLO và ViT
                    else:
                        cropped_object = frame[y1:y2, x1:x2]
                        if cropped_object.size > 0:
                            vit_label, vit_confidence = classify_with_vit(cropped_object, device)
                            if yolo_confidence + 0.2 > vit_confidence:
                                final_label = yolo_label  # YOLO tự tin hơn
                            else:
                                final_label = vit_label   # ViT tự tin hơn
                    # =========================================================

                    if final_label != 'other':
                        # Lưu nhãn và hộp giới hạn cho khung hình hiện tại
                        current_frame_objects.append({'label': final_label, 'box': (x1, y1, x2, y2)})

            frame_by_frame_results.append(current_frame_objects)

        # ==============================================================================
        # BƯỚC TRUNG GIAN ĐÃ BỊ LOẠI BỎ VÌ KHÔNG CÒN THEO DÕI
        # ==============================================================================

        # ==============================================================================
        # GIAI ĐOẠN 2: GHI VIDEO VÀ TỔNG HỢP KẾT QUẢ
        # ==============================================================================
        print("\nGiai đoạn 2: Ghi video và tổng hợp kết quả...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Tua lại video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_annotated_detection.mp4")
        out_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        video_frame_indices = {str(i): [] for i in range(1, 9)}
        video_question_frames = {str(i): {"start": None, "end": None} for i in range(1, 9)}

        for frame_idx in tqdm(range(total_frames), desc=f"Pass 2: Writing {video_name}"):
            ret, frame = cap.read()
            if not ret: break

            annotated_frame = frame.copy()
            objects_in_this_frame = frame_by_frame_results[frame_idx]
            frame_object_counts = defaultdict(int)

            for obj in objects_in_this_frame:
                box = obj['box']
                label = obj['label']

                frame_object_counts[label] += 1
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # THAY ĐỔI: Hiển thị nhãn, không còn ID
                label_text = f"{label}"
                cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                            2)

            def process_match(question_id):
                video_frame_indices[question_id].append(frame_idx)
                if video_question_frames[question_id]["start"] is None:
                    video_question_frames[question_id]["start"] = (frame_idx, annotated_frame.copy())
                video_question_frames[question_id]["end"] = (frame_idx, annotated_frame.copy())

            counts = frame_object_counts
            if counts['person'] >= 1 and counts['motorcycle'] >= 1: process_match('1')
            if counts['person'] >= 1 and counts['bicycle'] >= 1: process_match('2')
            if counts['car'] >= 1: process_match('3')
            if counts['person'] == 1 and counts['bicycle'] == 1: process_match('4')
            if counts['person'] >= 1 and counts['motorcycle'] >= 1 and counts['car'] >= 1: process_match('5')
            if counts['person'] > 1: process_match('6')
            if counts['motorcycle'] > 1: process_match('7')
            if counts['person'] == 3: process_match('8')

            out_writer.write(annotated_frame)

        cap.release()
        out_writer.release()

        for q_id, frames in video_question_frames.items():
            if frames["start"] is not None:
                start_idx, start_img = frames["start"]
                end_idx, end_img = frames["end"]
                cv2.imwrite(os.path.join(OUTPUT_FRAMES_DIR, f"{video_name}_frame{start_idx}_q{q_id}_start.jpg"),
                            start_img)
                if start_idx != end_idx:
                    cv2.imwrite(os.path.join(OUTPUT_FRAMES_DIR, f"{video_name}_frame{end_idx}_q{q_id}_end.jpg"),
                                end_img)

        for q_id, indices in video_frame_indices.items():
            if indices: final_results[q_id][video_name] = indices

        print(f"\nĐã xử lý xong video {video_name}. Video output được lưu tại: {output_video_path}")

    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    end_time = time.time()
    print(f"\n==============================================")
    print(f"XỬ LÝ HOÀN TẤT!")
    print(f"Tổng thời gian: {end_time - start_time:.2f} giây")
    print(f"Kết quả đã được lưu vào file: {OUTPUT_JSON_FILE}")
    print(f"Các frame trích xuất đã được lưu trong thư mục: {OUTPUT_FRAMES_DIR}")
    print(f"==============================================")


if __name__ == "__main__":
    analyze_videos_single_device()