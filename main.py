#!/usr/bin/env python3

import sys
import time
import cv2
from siamfc import TrackerSiamFC

def create_mil_tracker():
    for ctor_name in ["TrackerMIL_create", "legacy.TrackerMIL_create"]:
        parts = ctor_name.split(".")
        try:
            obj = cv2
            for p in parts:
                obj = getattr(obj, p)
            return obj()
        except Exception:
            continue
    raise RuntimeError(f"Could not create tracker MIL with this OpenCV build.")

def create_siam_tracker():
    net_path = 'pretrained/siamfc/model.pth'
    return TrackerSiamFC(net_path=net_path)

def open_capture(source: str):
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap
    return None

src = "video.mp4"

def main():
    cap = open_capture(src)
    if not cap.isOpened():
        print(f"[error] Could not open video source: {src}")
        sys.exit(1)

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[error] Could not read first frame from source.")
        sys.exit(1)

    print("[info] Select ROI on the displayed window. Press ENTER or SPACE when done, or c to cancel.")
    roi = cv2.selectROI("Select Object", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Object")

    try:
        tracker = create_siam_tracker()
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)

    init_bbox = roi
    init_frame = frame.copy()
    tracker.init(init_frame, init_bbox)
    print(f"[info] Initialized SiamFC tracker with bbox={init_bbox}")

    paused = False
    fps = 0.0

    window_name = "Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[info] End of stream or cannot read frame. Exiting.")
                break

            t0 = time.time()
            bbox = tracker.update(frame)
            t1 = time.time()

            frame_time = t1 - t0
            if frame_time > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / frame_time) if fps > 0 else (1.0 / frame_time)

            if bbox is not None:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Tracker: SiamFC", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Tracking failure", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[info] Quit requested.")
            break
        elif key == ord(" "):
            paused = not paused
            print(f"[info] Paused = {paused}")
            if paused:
                cv2.putText(frame, "PAUSED - press space to resume, 'r' to reselect ROI, 'q' to quit",
                            (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(window_name, frame)
        elif key == ord("r"):
            paused = True
            print("[info] Re-select ROI on current frame.")
            roi = cv2.selectROI("Re-select ROI", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Re-select ROI")
            if roi == (0, 0, 0, 0):
                print("[warn] Empty ROI selected; keeping current tracker.")
                paused = False
            else:
                try:
                    tracker = create_siam_tracker()
                    tracker.init(frame, roi)
                    print(f"[info] Reinitialized tracker with bbox={roi}")
                except Exception as e:
                    print(f"[error] Could not reinitialize tracker: {e}")
                paused = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
