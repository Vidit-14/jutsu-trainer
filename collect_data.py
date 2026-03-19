import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ── MediaPipe setup ───────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

options = HandLandmarkerOpts(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Signs ─────────────────────────────────────────────────────
SIGNS = ["rat", "ox", "tiger", "hare", "dragon",
         "snake", "horse", "ram", "monkey", "bird", "dog", "boar"]

IMG_SIZE        = 64
SAMPLES_PER_SIGN = 400
DATA_DIR        = "data"
PADDING         = 40   # pixels of padding around the hand crop

# Create folders
for sign in SIGNS:
    os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

def get_both_hands_bbox(result, frame_w, frame_h):
    """
    Returns a single bounding box that wraps ALL detected hands.
    This is the key — we treat both hands as one unit.
    """
    if not result.hand_landmarks:
        return None

    all_x, all_y = [], []
    for hand in result.hand_landmarks:
        for lm in hand:
            all_x.append(int(lm.x * frame_w))
            all_y.append(int(lm.y * frame_h))

    x1 = max(0,       min(all_x) - PADDING)
    y1 = max(0,       min(all_y) - PADDING)
    x2 = min(frame_w, max(all_x) + PADDING)
    y2 = min(frame_h, max(all_y) + PADDING)

    # Must be a reasonable size
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None

    return x1, y1, x2, y2

def crop_and_preprocess(frame, bbox):
    """Crop the bbox, convert to grayscale, resize to 64x64."""
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE for better contrast (helps with lighting variation)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return resized

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    current_idx  = 0
    collecting   = False
    sample_count = 0
    last_save    = 0
    SAVE_INTERVAL = 0.05  # save every 50ms to get varied samples

    print("\n=== CNN Hand Sign Data Collector ===")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE} grayscale")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print(f"\nFirst sign: {SIGNS[current_idx].upper()}")
    print("SPACE = start/stop  |  N = next sign  |  Q = quit\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            bbox = get_both_hands_bbox(result, w, h)

            # ── Collect ───────────────────────────────────────
            now = time.time()
            if collecting and bbox and (now - last_save) >= SAVE_INTERVAL:
                processed = crop_and_preprocess(frame, bbox)
                sign      = SIGNS[current_idx]
                # Count existing files to avoid overwriting
                existing  = len(os.listdir(os.path.join(DATA_DIR, sign)))
                filepath  = os.path.join(DATA_DIR, sign, f"{existing:04d}.jpg")
                cv2.imwrite(filepath, processed)
                sample_count += 1
                last_save     = now

                if sample_count >= SAMPLES_PER_SIGN:
                    collecting   = False
                    sample_count = 0
                    print(f"✅ Done: {sign.upper()}  ({SAMPLES_PER_SIGN} samples)")
                    if current_idx + 1 < len(SIGNS):
                        print(f"   Press N for next: {SIGNS[current_idx+1].upper()}")
                    else:
                        print("🎉 All signs collected! Run train_cnn.py next.")

            # ── Draw bbox on frame ────────────────────────────
            display = frame.copy()
            if bbox:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if collecting else (0, 165, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                # Show live preview of what gets saved
                x1p, y1p, x2p, y2p = bbox
                crop_prev = frame[y1p:y2p, x1p:x2p]
                if crop_prev.size > 0:
                    prev = cv2.resize(crop_prev, (128, 128))
                    display[10:138, w-138:w-10] = prev
                    cv2.rectangle(display, (w-138, 10), (w-10, 138),
                                  (255, 255, 255), 1)
                    cv2.putText(display, "preview",
                                (w-130, 155), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (200, 200, 200), 1)

            # ── HUD ───────────────────────────────────────────
            sign   = SIGNS[current_idx].upper()
            status = "● COLLECTING" if collecting else "○ READY"
            clr    = (0, 255, 0) if collecting else (0, 165, 255)
            hands_found = "✓ Hands detected" if bbox else "✗ No hands detected"
            hclr   = (0, 255, 0) if bbox else (0, 0, 255)

            cv2.rectangle(display, (0, 0), (440, 120), (0, 0, 0), -1)
            cv2.putText(display, f"Sign: {sign}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(display, status,
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, clr, 2)
            cv2.putText(display, f"Samples: {sample_count}/{SAMPLES_PER_SIGN}",
                        (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(display, hands_found,
                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hclr, 1)

            cv2.imshow("CNN Data Collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not bbox and not collecting:
                    print("⚠ No hands detected — show your hands first!")
                else:
                    collecting   = not collecting
                    sample_count = 0
                    print(f"{'▶ Started' if collecting else '⏸ Paused'} — {sign}")
            elif key == ord('n'):
                if current_idx + 1 < len(SIGNS):
                    current_idx += 1
                    collecting   = False
                    sample_count = 0
                    print(f"\n➡ Next: {SIGNS[current_idx].upper()}")
                else:
                    print("Already on last sign!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()