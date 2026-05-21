"""
MovementTracker — infers the target vehicle's apparent motion from bounding-box
history and builds a rolling log of human-readable direction signals.

Signals emitted per sample interval:
  LEFT / RIGHT    — horizontal drift in frame
  APPROACHING     — bbox growing  (car closing the gap / drone catching up)
  PULLING AWAY    — bbox shrinking (car opening the gap / drone falling behind)
  STEADY          — no significant movement on either axis
"""
from collections import deque
from typing import List

# Public signal constants
LEFT         = 'LEFT'
RIGHT        = 'RIGHT'
APPROACHING  = 'APPROACHING'
PULLING_AWAY = 'PULLING AWAY'
STEADY       = 'STEADY'

# Minimum change (normalised 0-1) to register a direction event
_HORIZ_THRESHOLD = 0.030   # fraction of frame width
_DEPTH_THRESHOLD = 0.025   # change in bbox-height / frame-height


class MovementTracker:
    """
    Call .update(bbox) every frame the target is locked.
    Call .get_log() to retrieve the rolling direction list (oldest → newest).
    Call .reset() when the target lock is lost or re-acquired.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        log_len: int = 12,      # max entries kept in the visible log
        sample_every: int = 8,  # emit a signal every N frames (~270 ms @ 30 fps)
    ):
        self.frame_w      = frame_w
        self.frame_h      = frame_h
        self.log_len      = log_len
        self.sample_every = sample_every

        # Raw position buffer — (normalised_cx, normalised_bbox_height)
        self._buf: deque = deque(maxlen=sample_every * 4)
        self._log: deque = deque(maxlen=log_len)
        self._tick = 0

    # ------------------------------------------------------------------
    def update(self, bbox) -> None:
        """Feed the current primary-target bbox (x1, y1, x2, y2) in pixels."""
        x1, y1, x2, y2 = bbox
        cx_n   = (x1 + x2) / 2.0 / self.frame_w
        size_n = max(0.0, y2 - y1) / self.frame_h
        self._buf.append((cx_n, size_n))
        self._tick += 1

        if self._tick % self.sample_every != 0:
            return
        if len(self._buf) < self.sample_every + 1:
            return

        prev = self._buf[-(self.sample_every + 1)]
        curr = self._buf[-1]
        dx   = curr[0] - prev[0]   # positive → moved right
        ds   = curr[1] - prev[1]   # positive → bbox grew (approaching)

        moved_h = abs(dx) >= _HORIZ_THRESHOLD
        moved_d = abs(ds) >= _DEPTH_THRESHOLD

        if moved_h:
            self._log.append(RIGHT if dx > 0 else LEFT)
        if moved_d:
            self._log.append(APPROACHING if ds > 0 else PULLING_AWAY)
        if not moved_h and not moved_d:
            self._log.append(STEADY)

    # ------------------------------------------------------------------
    def get_log(self) -> List[str]:
        """Return the rolling direction log, oldest first."""
        return list(self._log)

    def reset(self) -> None:
        self._buf.clear()
        self._log.clear()
        self._tick = 0
