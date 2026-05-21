from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import numpy as np


@dataclass
class DroneSignals:
    """
    All axes normalised to [-1.0, 1.0].
    A value of 0 means "hold position" on that axis.
    """
    yaw: float          # negative = rotate left,  positive = rotate right
    pitch: float        # negative = fly backward,  positive = fly forward
    throttle: float     # negative = descend,        positive = ascend
    roll: float         # negative = strafe left,    positive = strafe right (unused for basic follow)
    target_locked: bool
    target_id: Optional[int]
    distance_est: float  # 0.0 = far away, 1.0 = very close (bbox fills frame)
    target_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) pixels

    def to_dict(self):
        d = asdict(self)
        for k in ('yaw', 'pitch', 'throttle', 'roll', 'distance_est'):
            d[k] = round(d[k], 3)
        d.pop('target_bbox', None)   # internal — not needed in JSON output
        return d


_NO_SIGNAL = DroneSignals(0.0, 0.0, 0.0, 0.0, False, None, 0.0, None)


class DroneSignalGenerator:
    """
    Converts a list of DeepSort tracks into drone control signals.

    Control logic (proportional — replace gains with PID for a real drone):
      - Yaw:      steer so target is horizontally centred in frame
      - Throttle: adjust so target is vertically centred in frame
      - Pitch:    adjust so bbox height ≈ follow_ratio * frame height
      - Roll:     0 (not needed for a yaw-capable drone)
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        follow_ratio: float = 0.30,   # target bbox-height / frame-height to maintain
        deadzone: float = 0.05,
        yaw_gain: float = 1.2,
        throttle_gain: float = 0.8,
        pitch_gain: float = 2.0,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.follow_ratio = follow_ratio
        self.deadzone = deadzone
        self.yaw_gain = yaw_gain
        self.throttle_gain = throttle_gain
        self.pitch_gain = pitch_gain
        self.primary_id: Optional[int] = None

    def reset_lock(self):
        self.primary_id = None

    def _clamp(self, v: float) -> float:
        if abs(v) < self.deadzone:
            return 0.0
        return float(np.clip(v, -1.0, 1.0))

    def _select_target(self, tracks):
        confirmed = [t for t in tracks if t.is_confirmed()]
        if not confirmed:
            return None

        # Re-acquire existing lock first
        if self.primary_id is not None:
            for t in confirmed:
                if t.track_id == self.primary_id:
                    return t

        # Pick largest bounding box (closest vehicle)
        def area(t):
            x1, y1, x2, y2 = t.to_ltrb()
            return max(0, x2 - x1) * max(0, y2 - y1)

        best = max(confirmed, key=area)
        self.primary_id = best.track_id
        return best

    def compute(self, tracks) -> DroneSignals:
        target = self._select_target(tracks)
        if target is None:
            self.primary_id = None
            return _NO_SIGNAL

        x1, y1, x2, y2 = target.to_ltrb()
        bbox_cx = (x1 + x2) / 2.0
        bbox_cy = (y1 + y2) / 2.0
        bbox_h  = max(0.0, y2 - y1)

        # Normalised offsets from frame centre  (-1 … +1)
        h_err = (bbox_cx - self.frame_w / 2.0) / (self.frame_w / 2.0)
        v_err = (bbox_cy - self.frame_h / 2.0) / (self.frame_h / 2.0)

        # Distance error: positive → too close → pitch backward
        size_ratio   = bbox_h / self.frame_h
        distance_err = size_ratio - self.follow_ratio

        yaw      = self._clamp(h_err        * self.yaw_gain)
        throttle = self._clamp(-v_err       * self.throttle_gain)   # v_err>0 means car below centre → ascend
        pitch    = self._clamp(-distance_err * self.pitch_gain)      # too close → pitch back (negative)

        return DroneSignals(
            yaw=yaw,
            pitch=pitch,
            throttle=throttle,
            roll=0.0,
            target_locked=True,
            target_id=target.track_id,
            distance_est=float(np.clip(size_ratio, 0.0, 1.0)),
            target_bbox=(int(x1), int(y1), int(x2), int(y2)),
        )
