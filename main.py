import os
import cv2
import json
import glob
import time
from collections import defaultdict
from ultralytics import YOLO
from tqdm import tqdm

# ==============================================================================
# PHẦN CẤU HÌNH
# ==============================================================================
INPUT_VIDEO_DIR = "input_videos"
OUTPUT_VIDEO_DIR = "output_videos"
OUTPUT_FRAMES_DIR = "output_frames"
OUTPUT_JSON_FILE = "results.json"
MODEL_NAME = 'yolo12x.pt'  # Thay 'yolo12x.pt' bằng model của bạn


# ==============================================================================
# PHẦN XỬ LÝ CHÍNH (SINGLE-GPU/CPU)
# ==============================================================================

def analyze_videos_single_device():
    start_time = time.time()

    # 1. Tải mô hình
    print(f"Đang tải mô hình {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    CLASS_NAMES = model.names
    print("Mô hình đã được tải thành công.")

    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

    # 2. Lấy danh sách video
    video_paths = glob.glob(os.path.join(INPUT_VIDEO_DIR, '*'))
    if not video_paths:
        print(f"Không tìm thấy video nào trong thư mục '{INPUT_VIDEO_DIR}'.")
        return

    final_results = {str(i): {} for i in range(1, 9)}

    # 3. Xử lý tuần tự từng video
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n==============================================")
        print(f"Bắt đầu xử lý video: {video_name}")
        print(f"==============================================")

        if video_name in ['File_3', 'File_7', 'File_1']:
            current_confidence = 0.3
        else:
            current_confidence = 0.3
        print(f"Sử dụng confidence threshold cho video này: {current_confidence}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở file video {video_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_annotated.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # <<< NEW: We need two data structures per video >>>
        # 1. To store ALL frame indices for the JSON file
        video_frame_indices = {str(i): [] for i in range(1, 9)}
        # 2. To store ONLY the start/end frame image data for saving
        video_question_frames = {str(i): {"start": None, "end": None} for i in range(1, 9)}

        # Process the video frame by frame
        for frame_idx in tqdm(range(total_frames), desc=f"Processing {video_name}"):
            ret, frame = cap.read()
            if not ret: break

            results = model.predict(frame, conf=current_confidence, verbose=False)[0]
            annotated_frame = results.plot()

            frame_object_counts = defaultdict(int)
            if results.boxes is not None:
                for cls_id in results.boxes.cls:
                    class_name = CLASS_NAMES[int(cls_id)]
                    frame_object_counts[class_name] += 1

            # Helper function to process a match for a given question
            def process_match(question_id):
                # Action 1: Add the frame index to our list for the JSON
                video_frame_indices[question_id].append(frame_idx)

                # Action 2: Update the start/end frame data for image saving
                if video_question_frames[question_id]["start"] is None:
                    video_question_frames[question_id]["start"] = (frame_idx, annotated_frame.copy())
                video_question_frames[question_id]["end"] = (frame_idx, annotated_frame.copy())

            # Check all conditions and process matches
            if frame_object_counts['person'] >= 1 and frame_object_counts['motorcycle'] >= 1: process_match('1')
            if frame_object_counts['person'] >= 1 and frame_object_counts['bicycle'] >= 1: process_match('2')
            if frame_object_counts['car'] >= 1: process_match('3')
            if frame_object_counts['person'] == 1 and frame_object_counts['bicycle'] == 1: process_match('4')
            if (frame_object_counts['person'] >= 1 and frame_object_counts['motorcycle'] >= 1 and frame_object_counts[
                'car'] >= 1): process_match('5')
            if frame_object_counts['person'] > 1: process_match('6')
            if frame_object_counts['motorcycle'] > 1: process_match('7')
            if frame_object_counts['person'] == 3: process_match('8')

            out_writer.write(annotated_frame)

        cap.release()
        out_writer.release()

        # <<< NEW: Post-processing logic after the video is done >>>
        # 1. Save the start/end images to disk
        for q_id, frames in video_question_frames.items():
            if frames["start"] is not None:
                start_idx, start_img_data = frames["start"]
                end_idx, end_img_data = frames["end"]

                # Save the start frame
                start_filename = f"{video_name}_frame{start_idx}_q{q_id}_start.jpg"
                start_save_path = os.path.join(OUTPUT_FRAMES_DIR, start_filename)
                cv2.imwrite(start_save_path, start_img_data)

                # Save the end frame ONLY if it's a different frame from the start
                if start_idx != end_idx:
                    end_filename = f"{video_name}_frame{end_idx}_q{q_id}_end.jpg"
                    end_save_path = os.path.join(OUTPUT_FRAMES_DIR, end_filename)
                    cv2.imwrite(end_save_path, end_img_data)

        # 2. Update the final JSON results with the lists of frame indices
        for q_id, indices in video_frame_indices.items():
            if indices:  # If the list is not empty
                final_results[q_id][video_name] = indices

        print(f"Đã xử lý xong video {video_name}. Video output được lưu tại: {output_video_path}")

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