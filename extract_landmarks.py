"""
Extract pose, hand, and face landmarks from ASL reference clips using
MediaPipe Holistic.

Point this at a folder of clips named after their phrase (e.g. help.mp4,
bathroom.mp4, where_is_my_classroom.mp4) and it saves one JSON file per
clip containing per-frame landmark data, ready to be mapped onto your
VRM avatar's rig in the next step.

Usage:
    python3.11 extract_landmarks.py --input clips/ --output landmarks/
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic


def landmark_list_to_dict(landmark_list):
    """Convert a MediaPipe landmark list to plain dicts, or None if nothing was detected."""
    if landmark_list is None:
        return None
    return [
        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": getattr(lm, "visibility", None)}
        for lm in landmark_list.landmark
    ]


def extract_clip(video_path: Path, holistic) -> list:
    """Run Holistic over every frame of a clip and return a list of per-frame landmark dicts."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames_data = []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # MediaPipe expects RGB; OpenCV reads frames in BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        frames_data.append({
            "frame": frame_idx,
            "pose": landmark_list_to_dict(results.pose_landmarks),
            "left_hand": landmark_list_to_dict(results.left_hand_landmarks),
            "right_hand": landmark_list_to_dict(results.right_hand_landmarks),
            "face": landmark_list_to_dict(results.face_landmarks),
        })
        frame_idx += 1

    cap.release()
    return frames_data


def trim_to_hand_activity(frames_data: list) -> list:
    """Drop leading/trailing frames where neither hand was detected.

    These are almost always resting footage before or after the actual
    sign (hand at the signer's side), and MediaPipe's pose-only wrist
    estimate during those frames is unreliable since it has no hand
    landmarks to anchor to -- keeping them tends to drag the retargeted
    avatar's arm toward the body during what should be idle time.
    """
    def has_hand(fd):
        return fd["left_hand"] is not None or fd["right_hand"] is not None

    start = 0
    while start < len(frames_data) and not has_hand(frames_data[start]):
        start += 1

    end = len(frames_data)
    while end > start and not has_hand(frames_data[end - 1]):
        end -= 1

    return frames_data[start:end]


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks from ASL reference clips.")
    parser.add_argument("--input", required=True, help="Folder containing reference video clips")
    parser.add_argument("--output", required=True, help="Folder to save extracted landmark JSON files")
    parser.add_argument("--min-detection-confidence", type=float, default=0.3,
                         help="Lower = catches more borderline hand/pose detections, at the cost of some precision (default: 0.3)")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.3,
                         help="Lower = keeps tracking through faster/blurrier motion more often (default: 0.3)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    video_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in video_extensions)

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as holistic:
        for video_path in video_files:
            phrase_name = video_path.stem  # e.g. "help" from "help.mp4"
            print(f"Processing {video_path.name}...")

            try:
                frames_data = extract_clip(video_path, holistic)
            except RuntimeError as e:
                print(f"  Skipped: {e}")
                continue

            trimmed = trim_to_hand_activity(frames_data)
            if len(trimmed) < len(frames_data):
                print(f"  Trimmed {len(frames_data) - len(trimmed)} resting frames (no hand detected) from the ends")
            frames_data = trimmed

            out_path = output_dir / f"{phrase_name}.json"
            with open(out_path, "w") as f:
                json.dump(frames_data, f)

            detected_frames = sum(1 for fd in frames_data if fd["pose"] is not None)
            print(f"  Saved {len(frames_data)} frames ({detected_frames} with pose detected) -> {out_path}")


if __name__ == "__main__":
    main()