import cv2
import os


def extract_frames(video_path):
    """
    Opens a video file with OpenCV and extracts up to 8 evenly spaced frames.
    Saves each frame as frame_0.jpg, frame_1.jpg, etc. in deepguard-backend/.
    Returns a list of saved file paths.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    # Decide which frame indices to extract
    if total_frames <= 8:
        frame_indices = list(range(total_frames))
    else:
        # 8 evenly spaced: first, last, and 6 in between
        frame_indices = [
            int(round(i * (total_frames - 1) / 7)) for i in range(8)
        ]

    saved_paths = []

    for count, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            continue

        file_path = f"frame_{count}.jpg"
        cv2.imwrite(file_path, frame)
        saved_paths.append(file_path)

    cap.release()
    return saved_paths


def cleanup_files(file_paths):
    """
    Deletes each file in the list if it exists.
    Does not crash if a file is already deleted.
    """
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
