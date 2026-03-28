import os
import uuid
import cv2


def extract_frames(video_path: str, max_frames: int = 8) -> list:
    """
    Open a video file and extract up to max_frames frames
    evenly spaced across the entire video.
    Returns a list of saved JPEG file paths.
    """

    print(f"[Frames] Opening video: {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[Frames] ERROR: Could not open video file")
            return []

        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Frames] Total frames in video: {total_frames}")

        if total_frames == 0:
            print(f"[Frames] ERROR: Video has 0 frames")
            cap.release()
            return []

        # Decide how many frames to extract
        num_to_extract = min(max_frames, total_frames)
        print(f"[Frames] Will extract {num_to_extract} frames")

        # Calculate evenly spaced frame positions
        if num_to_extract == 1:
            frame_positions = [0]
        else:
            step = total_frames / num_to_extract
            frame_positions = [int(step * i) for i in range(num_to_extract)]

        print(f"[Frames] Frame positions: {frame_positions}")

        # Extract and save each frame
        saved_paths = []
        for i, pos in enumerate(frame_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            success, frame = cap.read()

            if not success:
                print(f"[Frames] WARNING: Could not read frame at position {pos}")
                continue

            # Save as temp JPEG file
            filename = f"temp_frame_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(filename, frame)
            saved_paths.append(filename)
            print(f"[Frames] Saved frame {i + 1}/{num_to_extract}: {filename}")

        cap.release()
        print(f"[Frames] Done! Extracted {len(saved_paths)} frames")
        return saved_paths

    except Exception as e:
        print(f"[Frames] ERROR: {e}")
        return []


def cleanup_files(file_paths: list):
    """
    Delete all files in the given list.
    Silently skips files that don't exist.
    """

    print(f"[Cleanup] Cleaning up {len(file_paths)} files...")

    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"[Cleanup] Deleted: {path}")
            else:
                print(f"[Cleanup] Skipped (not found): {path}")
        except Exception as e:
            print(f"[Cleanup] ERROR deleting {path}: {e}")

    print(f"[Cleanup] Done!")


# --- Quick test (only runs if you execute this file directly) ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_utils.py <path_to_video>")
        print("Example: python video_utils.py test_video.mp4")
    else:
        frames = extract_frames(sys.argv[1])
        print(f"\nExtracted frames: {frames}")

        if frames:
            answer = input("\nDelete extracted frames? (y/n): ")
            if answer.lower() == "y":
                cleanup_files(frames)
