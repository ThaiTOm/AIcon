# ==============================================================================
# PHẦN IMPORT THƯ VIỆN
# ==============================================================================
import os
import cv2
import json
import glob
import time
import shutil
from collections import defaultdict
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
CUSTOM_VIT_CLASSES = ['bike', 'motorbike']
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
    model = models.vit_b_16(weights=None)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cuda:0')))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("Mô hình ViT tùy chỉnh đã được tải thành công.")
    return model, preprocess


custom_vit_model, custom_vit_preprocess = load_custom_vit_model(
    CUSTOM_VIT_WEIGHTS_PATH,
    len(CUSTOM_VIT_CLASSES)
)


# ------------------------------------------------------------------------------
# 3. CÁC HÀM PHÂN LOẠI
# ------------------------------------------------------------------------------
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
            probabilities = torch.nn.functional.softmax(output, dim=1)
            max_prob, prediction_index_tensor = torch.max(probabilities, 1)
            prediction_index = prediction_index_tensor.item()
            confidence = max_prob.item()

        predicted_imagenet_label = imagenet_categories[prediction_index]
        custom_label = imagenet_to_custom_map.get(predicted_imagenet_label, "other")
        return custom_label, confidence
    except Exception:
        return "other", 0.0


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
            probabilities = torch.nn.functional.softmax(output, dim=1)
            max_prob, prediction_index_tensor = torch.max(probabilities, 1)
            prediction_index = prediction_index_tensor.item()
            confidence = max_prob.item()

        predicted_label = CUSTOM_VIT_CLASSES[prediction_index]
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
MODEL_NAME = 'yolo12x.pt'  # Using a standard model name

# Xóa các thư mục output cũ
for dir_path in [OUTPUT_FRAMES_DIR, OUTPUT_VIDEO_DIR]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


# ==============================================================================
# PHẦN XỬ LÝ CHÍNH
# ==============================================================================
def analyze_videos_single_device():
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")

    vit_model.to(device)
    if custom_vit_model:
        custom_vit_model.to(device)
    else:
        print("CẢNH BÁO: Không thể tải model ViT tùy chỉnh. Chức năng phân loại bike/motorbike sẽ không hoạt động.")
        return

    print(f"Đang tải mô hình {MODEL_NAME}...")
    yolo_model = YOLO(MODEL_NAME)
    CLASS_NAMES = yolo_model.names
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
        frame_by_frame_results = []

        # ==============================================================================
        # GIAI ĐOẠN 1: THEO DÕI, PHÂN LOẠI VÀ THU THẬP DỮ LIỆU
        # ==============================================================================
        print("Giai đoạn 1: Theo dõi, phân loại và thu thập dữ liệu...")

        # >>> THAY ĐỔI: Cấu trúc dữ liệu để lưu lịch sử của mỗi đối tượng được theo dõi
        # Định dạng: {track_id: [{'frame': int, 'label': str, 'confidence': float}, ...]}
        object_tracking_data = defaultdict(list)

        for frame_idx in tqdm(range(total_frames), desc=f"Pass 1: Tracking {video_name}"):
            ret, frame = cap.read()
            if not ret: break

            # >>> THAY ĐỔI: Sử dụng .track() thay vì .predict() để lấy tracker ID
            yolo_results = yolo_model.track(frame, conf=0.1, verbose=False, imgsz=1600, iou=0.15, persist=True, tracker="custom_track.yaml")[0]

            current_frame_objects = []
            # Kiểm tra xem kết quả có chứa hộp và ID theo dõi không
            if yolo_results.boxes is not None and yolo_results.boxes.id is not None:
                tracker_ids = yolo_results.boxes.id.int().cpu().tolist()

                for i, box in enumerate(yolo_results.boxes):
                    tracker_id = tracker_ids[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_label_id = int(box.cls[0])
                    yolo_label = CLASS_NAMES.get(yolo_label_id, "unknown")
                    yolo_confidence = float(box.conf[0])

                    final_label = "other"
                    final_confidence = 0.0  # >>> THÊM: Biến lưu xác suất cuối cùng

                    if yolo_label in ['motorcycle', 'bicycle']:
                        cropped_object = frame[y1:y2, x1:x2]
                        if cropped_object.size > 0:
                            final_label, final_confidence = classify_bike_motorbike_with_custom_vit(cropped_object,
                                                                                                    device)
                    else:
                        cropped_object = frame[y1:y2, x1:x2]
                        if cropped_object.size > 0:
                            vit_label, vit_confidence = classify_with_vit(cropped_object, device)
                            if yolo_confidence + 0.05 > vit_confidence:
                                final_label = yolo_label
                                final_confidence = yolo_confidence  # >>> LƯU XÁC SUẤT
                            else:
                                final_label = vit_label
                                final_confidence = vit_confidence  # >>> LƯU XÁC SUẤT

                    if final_label != 'other':
                        current_frame_objects.append({'label': final_label, 'box': (x1, y1, x2, y2)})
                        # >>> THAY ĐỔI: Lưu kết quả vào cấu trúc dữ liệu theo dõi
                        object_tracking_data[tracker_id].append({
                            'frame': frame_idx,
                            'label': final_label,
                            'confidence': final_confidence
                        })

            frame_by_frame_results.append(current_frame_objects)

        # ==============================================================================
        # >>> GIAI ĐOẠN 2 MỚI: PHÂN TÍCH DỮ LIỆU ĐỂ TÌM FRAME TỐT NHẤT
        # ==============================================================================
        print("\nGiai đoạn 2: Phân tích dữ liệu theo dõi...")
        best_frames_per_object = {}

        for track_id, data_points in object_tracking_data.items():
            if not data_points: continue

            # Kiểm tra xem lớp của đối tượng có nhất quán trong suốt quá trình theo dõi không
            first_label = data_points[0]['label']
            is_class_consistent = all(d['label'] == first_label for d in data_points)

            if is_class_consistent:
                # Nếu lớp nhất quán, tìm điểm dữ liệu (frame) có xác suất cao nhất
                best_point = max(data_points, key=lambda x: x['confidence'])
                best_frames_per_object[track_id] = {
                    'label': first_label,
                    'best_frame': best_point['frame'],
                    'highest_prob': best_point['confidence']
                }

        # In kết quả phân tích ra màn hình
        print("\n--- KẾT QUẢ PHÂN TÍCH FRAME CÓ XÁC SUẤT CAO NHẤT ---")
        if not best_frames_per_object:
            print("Không tìm thấy đối tượng nào có lớp nhất quán để phân tích.")
        else:
            for track_id, info in best_frames_per_object.items():
                print(
                    f"  - Đối tượng ID: {track_id}, Lớp: '{info['label']}'"
                    f"  -> Frame tốt nhất: {info['best_frame']}"
                    f" (Xác suất: {info['highest_prob']:.4f})"
                )
        print("-----------------------------------------------------------\n")

        # ==============================================================================
        # GIAI ĐOẠN 3: GHI VIDEO VÀ TỔNG HỢP KẾT QUẢ
        # ==============================================================================
        print("Giai đoạn 3: Ghi video và tổng hợp kết quả...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_annotated.mp4")
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
                label_text = f"{label}"
                cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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