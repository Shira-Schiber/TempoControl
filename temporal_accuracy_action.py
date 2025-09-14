import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import decord
from decord import VideoReader
import os
import json
import argparse
import pandas as pd
from collections import defaultdict

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
TARGET_SECONDS = 5

# ------------------------------------------------------------
# OPTICAL FLOW FUNCTION
# ------------------------------------------------------------
def most_motion_second(video_path: Path) -> int:
    vr = VideoReader(str(video_path), num_threads=1)
    fps = int(vr.get_avg_fps())
    total_frames = len(vr)

    sampling_rate = max(fps // 4, 1) 
    frame_indices = list(range(0, total_frames, sampling_rate))

    if len(frame_indices) < 2:
        return -1  # not enough frames

    frames = vr.get_batch(frame_indices).asnumpy()
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    flow_mags = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(mag.mean())

    # Divide into 5 seconds
    mags_per_sec = [[] for _ in range(TARGET_SECONDS)]
    for i, mag in enumerate(flow_mags):
        sec = min(i * sampling_rate // fps, TARGET_SECONDS - 1)
        mags_per_sec[sec].append(mag)

    avg_motion_per_sec = [np.mean(sec) if sec else 0 for sec in mags_per_sec]
    return int(np.argmax(avg_motion_per_sec))

def process_videos_from_csv(videos_path: str, csv_file: str) -> dict:
    df = pd.read_csv(csv_file)
    results = []
    per_second_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["prompt"]
        target_sec = int(row["mov_sec"])
        video_file = f"{prompt}-0.mp4"
        video_path = Path(videos_path) / video_file

        if not video_path.exists():
            print(f"❌ Missing video: {video_path}")
            continue
            # raise FileNotFoundError(f"Video file {video_file} not found in {videos_path}")

        predicted_sec = most_motion_second(video_path)
        is_correct = int(predicted_sec == target_sec)

        results.append({
            "prompt": prompt,
            "video_path": str(video_path),
            "target_second": target_sec,
            "predicted_second": predicted_sec,
            "correct": is_correct
        })

        per_second_stats[target_sec]["total"] += 1
        per_second_stats[target_sec]["correct"] += is_correct

    total_correct = sum(r["correct"] for r in results)
    total_videos = len(results)
    overall_accuracy = total_correct / total_videos if total_videos > 0 else 0

    per_second_accuracy = {
        sec: stat["correct"] / stat["total"] if stat["total"] > 0 else 0
        for sec, stat in per_second_stats.items()
    }

    return {
        "overall_accuracy": overall_accuracy,
        "per_second_accuracy": per_second_accuracy,
        "results": results
    }

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute optical flow accuracy per video")
    parser.add_argument("--output_path", help="Output JSON file path")
    parser.add_argument("--videos_path", required=True, help="Path to directory containing video files")
    parser.add_argument("--csv_file", required=True, help="CSV file with prompt and mov_sec columns")
    args = parser.parse_args()

    results = process_videos_from_csv(args.videos_path, args.csv_file)

    output_path = os.path.join(args.output_path, "temporal_accuracy_action.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()