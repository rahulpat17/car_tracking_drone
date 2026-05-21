import cv2
import numpy as np
from typing import List
from drone_signals import DroneSignals

# Direction chip colours (BGR)
_DIR_COLORS = {
    'LEFT':        (0,  220, 255),   # yellow
    'RIGHT':       (0,  220, 255),   # yellow
    'APPROACHING': (30,  80, 230),   # red-orange
    'PULLING AWAY':(210, 170,  30),  # cyan-blue
    'STEADY':      (130, 130, 130),  # grey
}


def _signal_bar(img, x, y, label: str, value: float, bar_w=160, bar_h=14):
    """Draw a labelled [-1, 1] signal bar."""
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (40, 40, 40), -1)
    mid = x + bar_w // 2
    cv2.line(img, (mid, y), (mid, y + bar_h), (90, 90, 90), 1)

    fill_x = mid + int(value * bar_w / 2)
    color = (50, 220, 80) if abs(value) < 0.3 else (0, 165, 255) if abs(value) < 0.7 else (30, 30, 255)
    if value >= 0:
        cv2.rectangle(img, (mid, y + 2), (max(mid, fill_x), y + bar_h - 2), color, -1)
    else:
        cv2.rectangle(img, (min(mid, fill_x), y + 2), (mid, y + bar_h - 2), color, -1)

    cv2.putText(img, f'{label}: {value:+.2f}', (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def _direction_chips(img, direction_log: List[str], frame_w: int, frame_h: int):
    """
    Draw the rolling direction log as coloured chips in the bottom-right corner.
    Most recent entry is at the bottom; oldest at the top.
    """
    if not direction_log:
        return

    chip_h   = 18
    chip_pad = 4
    font_scale = 0.42
    font_thick = 1
    margin_r = 10
    margin_b = 10

    # Measure chip widths
    chips = direction_log[-10:]   # show at most 10 recent entries
    widths = []
    for label in chips:
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        widths.append(tw + chip_pad * 2)

    total_h = len(chips) * (chip_h + 3) - 3
    panel_x = frame_w - max(widths) - margin_r - 4
    panel_y = frame_h - margin_b - total_h

    # Semi-transparent background panel
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (panel_x - 6, panel_y - 16),
                  (frame_w - margin_r + 2, frame_h - margin_b + 4),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Header label
    cv2.putText(img, 'VEHICLE MOVEMENT', (panel_x - 4, panel_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)

    for i, label in enumerate(chips):
        cy = panel_y + i * (chip_h + 3)
        color = _DIR_COLORS.get(label, (150, 150, 150))
        w = widths[i]

        # Chip background
        cv2.rectangle(img, (panel_x, cy), (panel_x + w, cy + chip_h), color, -1)
        # Chip text
        cv2.putText(img, label,
                    (panel_x + chip_pad, cy + chip_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (20, 20, 20), font_thick, cv2.LINE_AA)

        # Highlight the most recent entry with a white border
        if i == len(chips) - 1:
            cv2.rectangle(img, (panel_x, cy), (panel_x + w, cy + chip_h), (240, 240, 240), 1)


class Visualizer:
    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h

    def draw(self, frame, tracks, signals: DroneSignals,
             fps: float, direction_log: List[str] = None):
        out = frame.copy()
        cx, cy = self.fw // 2, self.fh // 2

        # Frame centre crosshair
        cv2.line(out, (cx - 25, cy), (cx + 25, cy), (0, 220, 220), 1)
        cv2.line(out, (cx, cy - 25), (cx, cy + 25), (0, 220, 220), 1)
        cv2.circle(out, (cx, cy), 4, (0, 220, 220), 1)

        # Draw bounding box for the PRIMARY track only (max 1 box)
        for track in tracks:
            if not track.is_confirmed():
                continue
            if track.track_id != signals.target_id:
                continue  # skip all non-primary tracks

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 60), 2)
            cv2.putText(out, f'ID {track.track_id}', (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 60), 1, cv2.LINE_AA)

            tcx, tcy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(out, (cx, cy), (tcx, tcy), (255, 165, 0), 1)
            cv2.circle(out, (tcx, tcy), 5, (0, 255, 255), -1)
            cv2.putText(out, f'dist_est: {signals.distance_est:.2f}',
                        (x1, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # Signal bars — bottom-left
        panel_x, panel_y = 10, self.fh - 120
        bars = [
            ('YAW     ', signals.yaw),
            ('PITCH   ', signals.pitch),
            ('THROTTLE', signals.throttle),
        ]
        overlay = out.copy()
        cv2.rectangle(overlay,
                      (panel_x - 4, panel_y - 18),
                      (panel_x + 178, panel_y + 90),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        for i, (label, val) in enumerate(bars):
            _signal_bar(out, panel_x, panel_y + i * 28, label, val)

        # Direction log — bottom-right
        if direction_log:
            _direction_chips(out, direction_log, self.fw, self.fh)

        # Status banner — top-left
        if signals.target_locked:
            banner = f'LOCKED  ID:{signals.target_id}'
            bcolor = (0, 255, 80)
        else:
            banner = 'SEARCHING...'
            bcolor = (0, 80, 255)
        cv2.putText(out, banner, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, bcolor, 2, cv2.LINE_AA)

        # FPS — top-right
        cv2.putText(out, f'{fps:.1f} fps', (self.fw - 90, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        return out
