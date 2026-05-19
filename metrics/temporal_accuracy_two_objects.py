import argparse
import pandas as pd
import os
import cv2
import numpy as np
from ultralytics import YOLOv10
from tqdm import tqdm
import json

def extract_evenly_spaced_frames(video_path, num_frames=16):
    cap = None
    frame = None
    frame_copy = None
    
    try:
        for _ in range(3):  # Try 3 times
            try:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(video_path)
                if cap is not None and cap.isOpened():
                    break
                print(f"Retrying to open video file: {video_path}")
            except Exception as e:
                print(f"Error opening video file (attempt {_+1}): {str(e)}")
                continue
        
        if cap is None or not cap.isOpened():
            print(f"Error: Could not open video file after retries: {video_path}")
            return []

        # Get video properties
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                print(f"Error: Video has 0 frames: {video_path}")
                return []
        except Exception as e:
            print(f"Error getting frame count: {str(e)}")
            return []

        # Calculate target frame indices
        target_indices = np.linspace(0, total_frames-1, num=min(num_frames, total_frames), dtype=int)
        frames = []
        current_frame_idx = 0
        frame_read_count = 0

        # Read frames sequentially 
        while frame_read_count < total_frames and len(frames) < num_frames:
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if current_frame_idx in target_indices:
                    try:
                        frame_copy = frame.copy()
                        
                        if frame_copy is not None and frame_copy.size > 0:
                            frames.append(frame_copy)
                        else:
                            print(f"Warning: Invalid frame {current_frame_idx} from {video_path}")
                    except Exception as e:
                        print(f"Error copying frame {current_frame_idx}: {str(e)}")
                        continue
                        
                current_frame_idx += 1
                frame_read_count += 1

                # Free some memory periodically
                if frame_read_count % 30 == 0:
                    if frame is not None:
                        del frame
                        frame = None
                    if frame_copy is not None:
                        del frame_copy
                        frame_copy = None
                    
            except Exception as e:
                print(f"Error reading frame {current_frame_idx} from {video_path}: {str(e)}")
                break

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return []
    
    finally:
        # Clean up
        try:
            if cap is not None:
                cap.release()
            if frame is not None:
                del frame
            if frame_copy is not None:
                del frame_copy
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    if not frames:
        print(f"Warning: No valid frames extracted from {video_path}")
    
    return frames

def parse_frame_config(config_str):
    """Parse frame configuration string from CSV into a list of integers"""
    try:
        # Remove any extra whitespace and split
        config_values = config_str.strip().split()
        # Convert to integers
        return [int(val) for val in config_values]
    except Exception as e:
        print(f"Error parsing frame config: {str(e)}")
        return None

def main(args):
    # Find the CSV file in the videos_path directory
    csv = pd.read_csv(args.csv_file)
    model = YOLOv10.from_pretrained('jameslahm/yolov10x')
    results_list = []

    total_static_success = 0
    total_temp_success = 0
    total_static_frames = 0
    total_temp_frames = 0
    for idx, row in tqdm(csv.iterrows(), total=len(csv)):
        prompt = row['original_prompt']
        temp_obj = str(row['temp_object']).strip().lower()
        static_obj = str(row['static_object']).strip().lower()
        frame_config = parse_frame_config(row['control_signal1'])

        video_results_for_prompt = []
        missing_files = []

       
        for i in range(1):
            video_name = f"{prompt}-{i}.mp4"
            video_path = os.path.join(args.videos_path, video_name)
            if not os.path.isfile(video_path):
                missing_files.append(video_path)
                continue

            frames = extract_evenly_spaced_frames(video_path, len(frame_config))
            frame_count = len(frames)
            success_count = 0
            static_success = 0
            temp_success = 0
            for j, frame in enumerate(frames):
                output = model.predict(source=frame, save=False)
                result = output[0]
                detected_names = [result.names[int(k)].lower() for k in result.boxes.cls.cpu().numpy()]
                # Use video-specific frame_config to determine success condition
                if frame_config[j] == 0:
                    # Only static object, temp object must NOT appear
                    if static_obj in detected_names and temp_obj not in detected_names:
                        success_count += 1
                        static_success += 1
                else:
                    # Both static and temp object must appear
                    if static_obj in detected_names and temp_obj in detected_names:
                        success_count += 1
                        temp_success += 1
            video_result = success_count / frame_count if frame_count > 0 else 0.0
            static_rate = static_success / frame_config.count(0) if frame_config.count(0) > 0 else 0.0
            temp_rate = temp_success / frame_config.count(1) if frame_config.count(1) > 0 else 0.0
            video_results_for_prompt.append({
                "video_path": video_path,
                "video_results": video_result,
                "success_frame_count": success_count,
                "frame_count": frame_count,
                "static_object_success_rate": static_rate,
                "temp_object_success_rate": temp_rate
            })
            total_static_success += static_success
            total_temp_success += temp_success
            total_static_frames += frame_config.count(0)
            total_temp_frames += frame_config.count(1)

        if missing_files:
            print(f"Warning: For prompt '{prompt}', the following video files were not found:")
            for mf in missing_files:
                print(f"   - {mf}")

        if not video_results_for_prompt:
            video_results_for_prompt.append({
                "video_path": None,
                "video_results": 0.0,
                "success_frame_count": 0,
                "frame_count": 0,
                "static_object_success_rate": 0.0,
                "temp_object_success_rate": 0.0
            })

        results_list.extend(video_results_for_prompt)

    mean_metric = (
        np.mean([r["video_results"] for r in results_list if r["frame_count"] > 0])
        if results_list else 0.0
    )
    global_static_rate = total_static_success / total_static_frames if total_static_frames > 0 else 0.0
    global_temp_rate = total_temp_success / total_temp_frames if total_temp_frames > 0 else 0.0

    final_json = {
        "temporal_accuracy": [
            mean_metric,
            results_list
        ],
        "global_absent_object_success_rate": global_static_rate,
        "global_present_object_success_rate": global_temp_rate
    }

    output_json = os.path.join(args.output_path, 'temporal_accuracy_two_objects.json')
    with open(output_json, "w") as f:
        json.dump(final_json, f, indent=2)

    print(f"Saved temporal accuracy two objects metric results to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output JSON')
    parser.add_argument('--videos_path', type=str, required=True, help='Path to directory containing videos and CSV')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing video metadata')

    args = parser.parse_args()
    main(args)