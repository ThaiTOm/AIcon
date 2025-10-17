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
import numpy as np
from torchvision.models import ViT_B_16_Weights, vit_b_16
from PIL import Image

# ==============================================================================
# PHẦN CẤU HÌNH VÀ KHỞI TẠO MODEL
# ==============================================================================

# 1. Định nghĩa ánh xạ từ ImageNet sang các lớp tùy chỉnh
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

# 2. Tải mô hình ViT (chỉ một lần)
print("Đang tải mô hình Vision Transformer (ViT)...")
vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
vit_model = vit_b_16(weights=vit_weights)
vit_model.eval()
vit_preprocess = vit_weights.transforms()
imagenet_categories = vit_weights.meta["categories"]
print("Mô hình ViT đã được tải.")


# 3. Hàm phân loại bằng ViT
def classify_with_vit(image_crop_np, device='cpu'):
    try:
        img_rgb = cv2.cvtColor(image_crop_np, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = vit_preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = vit_model(img_tensor)
        prediction_index = output.argmax(dim=1)[0]
        predicted_imagenet_label = imagenet_categories[prediction_index]
        return imagenet_to_custom_map.get(predicted_imagenet_label, "other")
    except Exception:
        return "other"


# ==============================================================================
# PHẦN CẤU HÌNH
# ==============================================================================
INPUT_VIDEO_DIR = "input_videos"
OUTPUT_VIDEO_DIR = "output_videos"
OUTPUT_FRAMES_DIR = "output_frames"
# OUTPUT_DIFFERENCES_DIR đã bị loại bỏ vì không còn phù hợp
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
    vit_model.to(device)

    # 1. Tải mô hình YOLO
    print(f"Đang tải mô hình {MODEL_NAME}...")
    yolo_model = YOLO(MODEL_NAME)
    CLASS_NAMES = yolo_model.names
    print("Mô hình YOLO đã được tải thành công.")

    # Thêm các nhãn gốc của YOLO vào map để xử lý nhất quán
    for key in CLASS_NAMES.values():
        if key in CUSTOM_CLASSES:
            imagenet_to_custom_map[key] = key

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

        # --- CẤU TRÚC LƯU TRỮ DỮ LIỆU ---
        tracked_object_classifications = defaultdict(list)
        frame_by_frame_results = []

        # ==============================================================================
        # GIAI ĐOẠN 1: THEO DÕI, PHÂN LOẠI VÀ THU THẬP DỮ LIỆU
        # ==============================================================================
        print("Giai đoạn 1: Theo dõi và thu thập dữ liệu...")
        for frame_idx in tqdm(range(total_frames), desc=f"Pass 1: Tracking {video_name}"):
            ret, frame = cap.read()
            if not ret: break

            yolo_results = yolo_model.track(frame, conf=0.4, persist=True, verbose=False)[0]

            current_frame_objects = []
            if yolo_results.boxes is not None and yolo_results.boxes.id is not None:
                for box in yolo_results.boxes:
                    tracker_id = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_label = CLASS_NAMES[int(box.cls[0])]

                    final_label = "other"
                    if yolo_label == 'person':
                        final_label = 'person'
                    else:
                        cropped_object = frame[y1:y2, x1:x2]
                        if cropped_object.size > 0:
                            final_label = classify_with_vit(cropped_object, device)

                    if final_label != 'other':
                        tracked_object_classifications[tracker_id].append(final_label)
                        current_frame_objects.append({'id': tracker_id, 'box': (x1, y1, x2, y2)})

            frame_by_frame_results.append(current_frame_objects)

        # ==============================================================================
        # BƯỚC TRUNG GIAN: QUYẾT ĐỊNH NHÃN CUỐI CÙNG CHO MỖI ĐỐI TƯỢNG
        # ==============================================================================
        print("\nBước trung gian: Quyết định nhãn cuối cùng...")
        final_object_labels = {}
        for tracker_id, labels in tracked_object_classifications.items():
            if labels:
                # Tìm nhãn xuất hiện nhiều nhất
                most_common_label = Counter(labels).most_common(1)[0][0]
                final_object_labels[tracker_id] = most_common_label

        # ==============================================================================
        # GIAI ĐOẠN 2: GHI VIDEO VÀ KẾT QUẢ VỚI NHÃN CUỐI CÙNG
        # ==============================================================================
        print("Giai đoạn 2: Ghi video và tổng hợp kết quả...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Tua lại video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_annotated_final.mp4")
        out_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        video_frame_indices = {str(i): [] for i in range(1, 9)}
        video_question_frames = {str(i): {"start": None, "end": None} for i in range(1, 9)}

        for frame_idx in tqdm(range(total_frames), desc=f"Pass 2: Writing {video_name}"):
            ret, frame = cap.read()
            if not ret: break

            annotated_frame = frame.copy()
            objects_in_this_frame = frame_by_frame_results[frame_idx]
            frame_object_counts_final = defaultdict(int)

            for obj in objects_in_this_frame:
                tracker_id = obj['id']
                box = obj['box']
                final_label = final_object_labels.get(tracker_id)

                if final_label:
                    frame_object_counts_final[final_label] += 1
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"ID {tracker_id}: {final_label}"
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                2)

            # --- KIỂM TRA ĐIỀU KIỆN DỰA TRÊN KẾT QUẢ CUỐI CÙNG ---
            def process_match(question_id):
                video_frame_indices[question_id].append(frame_idx)
                if video_question_frames[question_id]["start"] is None:
                    video_question_frames[question_id]["start"] = (frame_idx, annotated_frame.copy())
                video_question_frames[question_id]["end"] = (frame_idx, annotated_frame.copy())

            counts = frame_object_counts_final
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

        # Lưu các frame đầu/cuối và kết quả JSON
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

    # 5. Lưu kết quả cuối cùng
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