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
import multiprocessing as mp

# ==============================================================================
# PHẦN CẤU HÌNH VÀ KHỞI TẠO MODEL (SẼ ĐƯỢC GỌI BÊN TRONG WORKER)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. CẤU HÌNH MODEL ViT CHUNG (PRE-TRAINED TRÊN IMAGENET)
# ------------------------------------------------------------------------------
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
imagenet_to_custom_map = {label: custom_class for custom_class, labels in CUSTOM_CLASSES.items() for label in labels}

# ------------------------------------------------------------------------------
# 2. CẤU HÌNH MODEL ViT TÙY CHỈNH
# ------------------------------------------------------------------------------
CUSTOM_VIT_WEIGHTS_PATH = "bike_motorbike_vit_weights.pth"
CUSTOM_VIT_CLASSES = ['bike', 'motorbike']
CUSTOM_VIT_LABEL_MAP = {
    'bike': 'bicycle',
    'motorbike': 'motorcycle'
}


def load_custom_vit_model(weights_path, num_classes, device):
    if not os.path.exists(weights_path):
        print(f"LỖI [GPU {device}]: Không tìm thấy tệp trọng số '{weights_path}'.")
        return None, None
    print(f"GPU {device}: Đang tải mô hình ViT tùy chỉnh từ '{weights_path}'...")
    model = models.vit_b_16(weights=None)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print(f"GPU {device}: Mô hình ViT tùy chỉnh đã được tải thành công.")
    return model, preprocess


# ------------------------------------------------------------------------------
# 3. CÁC HÀM PHÂN LOẠI
# ------------------------------------------------------------------------------
def classify_with_vit(image_crop_np, vit_model, vit_preprocess, imagenet_categories, device):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image_crop_np, cv2.COLOR_BGR_RGB))
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


def classify_bike_motorbike_with_custom_vit(image_crop_np, custom_vit_model, custom_vit_preprocess, device):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image_crop_np, cv2.COLOR_BGR_RGB))
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
OUTPUT_JSON_FILE = "results.json"
MODEL_NAME = 'yolo12x.pt'


# ==============================================================================
# PHẦN XỬ LÝ CHÍNH - WORKER FUNCTION FOR EACH GPU
# ==============================================================================
def process_videos_on_device(video_paths_subset, device_id, results_queue):
    device = f'cuda:{device_id}'
    torch.cuda.set_device(device)
    print(f"Worker process {os.getpid()} started, assigned to device: {device}")

    # TẢI MODEL BÊN TRONG WORKER
    print(f"GPU {device}: Đang tải mô hình Vision Transformer (ViT) từ ImageNet...")
    vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    vit_model = models.vit_b_16(weights=vit_weights)
    vit_model.eval()
    vit_model.to(device)
    vit_preprocess = vit_weights.transforms()
    imagenet_categories = vit_weights.meta["categories"]

    custom_vit_model, custom_vit_preprocess = load_custom_vit_model(
        CUSTOM_VIT_WEIGHTS_PATH, len(CUSTOM_VIT_CLASSES), device
    )
    if custom_vit_model:
        custom_vit_model.to(device)
    else:
        print(f"CẢNH BÁO [GPU {device}]: Không thể tải model ViT tùy chỉnh.")
        return

    print(f"GPU {device}: Đang tải mô hình {MODEL_NAME}...")
    yolo_model = YOLO(MODEL_NAME)
    yolo_model.to(device)
    CLASS_NAMES = yolo_model.names

    worker_results = {"question_1": {}, "question_2": {}, "question_3": {}}

    for video_path in video_paths_subset:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nGPU {device}: Bắt đầu xử lý video: {video_name}")

        cap = cv2.VideoCapture(video_path)
        ### THAY ĐỔI: Lấy thông số video để sử dụng sau này ###
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # GIAI ĐOẠN 1: THU THẬP DỮ LIỆU TỪ VIDEO
        print(f"GPU {device}: Giai đoạn 1: Thu thập dữ liệu từ {video_name}...")

        ### THAY ĐỔI: Lưu trữ thông tin chi tiết của đối tượng, không chỉ số lượng ###
        frame_by_frame_detections = []

        for frame_idx in tqdm(range(total_frames), desc=f"GPU {device} Pass 1: Analyzing {video_name}"):
            ret, frame = cap.read()
            if not ret: break

            yolo_results = yolo_model.track(frame, conf=0.1, verbose=False, imgsz=1600, iou=0.15, persist=True,
                                            tracker="custom_track.yaml")[0]

            current_frame_detections = []

            if yolo_results.boxes is not None and yolo_results.boxes.id is not None:
                for i, box in enumerate(yolo_results.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_label_id = int(box.cls[0])
                    yolo_label = CLASS_NAMES.get(yolo_label_id, "unknown")
                    yolo_confidence = float(box.conf[0])

                    final_label = "other"
                    cropped_object = frame[y1:y2, x1:x2]
                    if cropped_object.size <= 0: continue

                    if yolo_label in ['motorcycle', 'bicycle']:
                        final_label, _ = classify_bike_motorbike_with_custom_vit(cropped_object, custom_vit_model,
                                                                                 custom_vit_preprocess, device)
                    else:
                        vit_label, vit_confidence = classify_with_vit(cropped_object, vit_model, vit_preprocess,
                                                                      imagenet_categories, device)
                        final_label = yolo_label if yolo_confidence + 0.05 > vit_confidence else vit_label

                    if final_label != 'other':
                        ### THAY ĐỔI: Lưu nhãn và hộp giới hạn ###
                        current_frame_detections.append({'label': final_label, 'box': (x1, y1, x2, y2)})

            frame_by_frame_detections.append(current_frame_detections)

        # GIAI ĐOẠN 2: PHÂN TÍCH DỮ LIỆU ĐÃ THU THẬP ĐỂ TRẢ LỜI CÂU HỎI
        print(f"GPU {device}: Giai đoạn 2: Phân tích dữ liệu cho {video_name}...")

        q1_state, q1_one_car_frames, q1_two_car_frames = 0, [], []
        q2_state, q2_person_bicycle_frames, q2_all_three_frames = 0, [], []
        q3_is_video_valid, q3_target_frames = True, []

        ### THAY ĐỔI: Lặp qua dữ liệu chi tiết và tạo 'counts' một cách linh hoạt ###
        for frame_idx, detections in enumerate(frame_by_frame_detections):
            counts = defaultdict(int)
            for det in detections:
                counts[det['label']] += 1
            total_objects = len(detections)

            # --- Logic xử lý cho các câu hỏi (giữ nguyên) ---
            # Câu hỏi 1
            if q1_state < 3:
                num_cars = counts['car']
                if q1_state == 0 and num_cars == 1:
                    q1_state = 1; q1_one_car_frames.append(frame_idx)
                elif q1_state == 1:
                    if num_cars == 1 and len(q1_one_car_frames) < 10:
                        q1_one_car_frames.append(frame_idx)
                    elif num_cars == 2:
                        q1_state = 2; q1_two_car_frames.append(frame_idx)
                    elif num_cars != 1:
                        q1_state = 0; q1_one_car_frames.clear()
                elif q1_state == 2:
                    if num_cars == 2 and len(q1_two_car_frames) < 10:
                        q1_two_car_frames.append(frame_idx)
                    elif num_cars == 3:
                        q1_state = 3
                    elif num_cars != 2:
                        q1_state = 0; q1_one_car_frames.clear(); q1_two_car_frames.clear()
            # Câu hỏi 2
            if q2_state < 3:
                only_person = counts['person'] >= 1 and total_objects == counts['person']
                only_person_bicycle = counts['person'] >= 1 and counts['bicycle'] >= 1 and total_objects == (
                            counts['person'] + counts['bicycle'])
                only_person_bicycle_car = counts['person'] >= 1 and counts['bicycle'] >= 1 and counts[
                    'car'] >= 1 and total_objects == (counts['person'] + counts['bicycle'] + counts['car'])
                if q2_state == 0 and only_person: q2_state = 1
                if q2_state == 1 and only_person_bicycle:
                    q2_state = 2; q2_person_bicycle_frames.append(frame_idx)
                elif q2_state == 2:
                    if only_person_bicycle:
                        q2_person_bicycle_frames.append(frame_idx)
                    elif only_person_bicycle_car:
                        q2_state = 3; q2_all_three_frames.append(frame_idx)
                elif q2_state == 3 and only_person_bicycle_car:
                    q2_all_three_frames.append(frame_idx)
            # Câu hỏi 3
            if q3_is_video_valid:
                if total_objects != (counts['person'] + counts['motorcycle']) or counts['person'] > 3 or counts[
                    'motorcycle'] > 3:
                    q3_is_video_valid = False
                elif counts['person'] == 3 and counts['motorcycle'] == 3 and len(q3_target_frames) < 10:
                    q3_target_frames.append(frame_idx)

        # --- TỔNG HỢP KẾT QUẢ JSON ---
        if q1_state == 3: worker_results["question_1"][video_name] = {"ten_frames_with_one_car": q1_one_car_frames,
                                                                      "ten_frames_with_two_cars": q1_two_car_frames}
        if q2_state == 3 and q2_all_three_frames: worker_results["question_2"][video_name] = {
            "frames_with_person_and_bicycle_only": q2_person_bicycle_frames,
            "frames_with_all_three_objects_only": q2_all_three_frames}
        if q3_is_video_valid and q3_target_frames: worker_results["question_3"][video_name] = {
            "ten_frames_with_3_persons_and_3_motorcycles": q3_target_frames}

        ### THAY ĐỔI: GIAI ĐOẠN 3 - GHI VIDEO OUTPUT ###
        print(f"GPU {device}: Giai đoạn 3: Ghi video output cho {video_name}...")
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_annotated.mp4")
        out_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Tua video về đầu

        for frame_idx in tqdm(range(total_frames), desc=f"GPU {device} Pass 2: Writing {video_name}"):
            ret, frame = cap.read()
            if not ret: break

            annotated_frame = frame.copy()
            detections_for_this_frame = frame_by_frame_detections[frame_idx]

            for det in detections_for_this_frame:
                label = det['label']
                x1, y1, x2, y2 = det['box']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out_writer.write(annotated_frame)

        out_writer.release()
        print(f"GPU {device}: Đã lưu video output tại: {output_video_path}")

        cap.release()
        print(f"GPU {device}: Đã phân tích xong video {video_name}.")

    results_queue.put(worker_results)
    print(f"Worker process {os.getpid()} on device {device} finished.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    start_time = time.time()

    if os.path.exists(OUTPUT_VIDEO_DIR):
        shutil.rmtree(OUTPUT_VIDEO_DIR)
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Không tìm thấy GPU. Vui lòng chạy trên máy có GPU NVIDIA và cài đặt CUDA.")
        exit()
    print(f"Tìm thấy {num_gpus} GPU.")

    all_video_paths = glob.glob(os.path.join(INPUT_VIDEO_DIR, '*'))
    if not all_video_paths:
        print(f"Không tìm thấy video nào trong thư mục '{INPUT_VIDEO_DIR}'.")
        exit()

    videos_per_gpu = [[] for _ in range(num_gpus)]
    for i, video_path in enumerate(all_video_paths):
        videos_per_gpu[i % num_gpus].append(video_path)

    manager = mp.Manager()
    results_queue = manager.Queue()

    processes = []
    for i in range(num_gpus):
        if not videos_per_gpu[i]: continue
        p = mp.Process(target=process_videos_on_device, args=(videos_per_gpu[i], i, results_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\nĐang tổng hợp kết quả từ tất cả các worker...")
    final_results = {"question_1": {}, "question_2": {}, "question_3": {}}
    while not results_queue.empty():
        worker_result = results_queue.get()
        for q_id, video_data in worker_result.items():
            final_results[q_id].update(video_data)

    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    end_time = time.time()
    print(f"\n==============================================")
    print(f"XỬ LÝ HOÀN TẤT!")
    print(f"Tổng thời gian xử lý {len(all_video_paths)} video trên {num_gpus} GPU: {end_time - start_time:.2f} giây")
    print(f"Kết quả đã được lưu vào file: {OUTPUT_JSON_FILE}")
    print(f"Các video đã xử lý được lưu trong thư mục: {OUTPUT_VIDEO_DIR}")
    print(f"==============================================")