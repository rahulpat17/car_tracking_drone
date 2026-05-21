"""
Car-tracking drone signal system
─────────────────────────────────
Webcam (default):
    python main.py

Screen capture — primary monitor:
    python main.py --source screen

Screen capture — secondary monitor:
    python main.py --source screen --monitor 2

Second physical camera:
    python main.py --camera 1

Larger / more accurate model:
    python main.py --model s

Print JSON signals to stdout each frame:
    python main.py --print-signals

Tune following distance or detection sensitivity:
    python main.py --follow-ratio 0.35 --conf 0.35
"""
import argparse
import json
import time
from collections import deque

import cv2

from capture import make_capture
from drone_signals import DroneSignalGenerator
from movement import MovementTracker
from tracker import VehicleTracker
from visualizer import Visualizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source',        default='camera',
                   choices=['camera', 'screen'],
                   help='Input source: webcam or live screen capture')
    p.add_argument('--camera',        type=int,   default=0,
                   help='Camera device index (used with --source camera)')
    p.add_argument('--monitor',       type=int,   default=1,
                   help='Monitor index for screen capture: 1=primary, 2=secondary …')
    p.add_argument('--model',         default='n', choices=['n', 's', 'm'],
                   help='YOLOv8 size: n(ano) · s(mall) · m(edium)')
    p.add_argument('--conf',          type=float, default=0.40,
                   help='Detection confidence threshold 0–1')
    p.add_argument('--follow-ratio',  type=float, default=0.30,
                   help='Target bbox-height / frame-height to maintain')
    p.add_argument('--width',         type=int,   default=1280)
    p.add_argument('--height',        type=int,   default=720)
    p.add_argument('--print-signals', action='store_true',
                   help='Print drone signals + direction log as JSON each frame')
    return p.parse_args()


def main():
    args = parse_args()

    cap = make_capture(
        source=args.source,
        camera=args.camera,
        monitor=args.monitor,
        width=args.width,
        height=args.height,
    )
    fw, fh = cap.width, cap.height
    print(f"Resolution: {fw}x{fh}  |  source: {args.source}")

    tracker    = VehicleTracker(model_size=args.model, conf=args.conf)
    signal_gen = DroneSignalGenerator(fw, fh, follow_ratio=args.follow_ratio)
    move_track = MovementTracker(fw, fh)
    viz        = Visualizer(fw, fh)

    fps_buf = deque(maxlen=30)
    t_prev  = time.perf_counter()

    print("\nRunning — controls:")
    print("  q  quit")
    print("  r  reset target lock")
    print("  +  raise confidence threshold")
    print("  -  lower confidence threshold\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed.")
            break

        tracks  = tracker.update(frame)
        signals = signal_gen.compute(tracks)

        # Update movement tracker only while target is locked
        if signals.target_locked and signals.target_bbox is not None:
            move_track.update(signals.target_bbox)
        elif not signals.target_locked:
            move_track.reset()

        direction_log = move_track.get_log()

        # FPS
        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        fps = sum(fps_buf) / len(fps_buf)

        display = viz.draw(frame, tracks, signals, fps, direction_log)
        cv2.imshow('Car Tracker — Drone Signals', display)

        if args.print_signals:
            out = signals.to_dict()
            out['direction_log'] = direction_log
            print(json.dumps(out), flush=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            signal_gen.reset_lock()
            move_track.reset()
            print("Target lock reset.")
        elif key == ord('+'):
            tracker.conf = min(tracker.conf + 0.05, 0.95)
            print(f"Confidence → {tracker.conf:.2f}")
        elif key == ord('-'):
            tracker.conf = max(tracker.conf - 0.05, 0.10)
            print(f"Confidence → {tracker.conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
