"""
Unified capture interface — same .read() / .release() API for both sources.

Sources:
  CameraCapture  — physical webcam or USB camera
  ScreenCapture  — live screen grab using mss (no second laptop needed)

Factory:
  make_capture(source, ...)  — returns the right object based on 'camera'|'screen'
"""
import cv2
import numpy as np


class CameraCapture:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index}. "
                "Try --camera 1 or --camera 2."
            )
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


class ScreenCapture:
    """
    Grabs a full monitor and resizes it to (width, height) before returning.
    Monitor indices: 1 = primary, 2 = secondary, etc.
    (Index 0 is the virtual all-monitors bounding box — usually not what you want.)
    """
    def __init__(self, monitor: int = 1, width: int = 1280, height: int = 720):
        try:
            import mss as _mss
        except ImportError:
            raise ImportError(
                "mss is required for screen capture. Run: pip install mss"
            )
        self._mss = _mss.mss()
        monitors = self._mss.monitors
        print(f"Available monitors: {len(monitors) - 1}  "
              f"(indices 1–{len(monitors) - 1})")
        for i, m in enumerate(monitors[1:], start=1):
            print(f"  Monitor {i}: {m['width']}x{m['height']} "
                  f"at ({m['left']}, {m['top']})")
        if monitor < 1 or monitor >= len(monitors):
            raise RuntimeError(
                f"Monitor {monitor} not found. "
                f"Valid range: 1–{len(monitors) - 1}"
            )
        self._mon  = monitors[monitor]
        self.width  = width
        self.height = height
        print(f"Screen capture → monitor {monitor}  "
              f"({self._mon['width']}x{self._mon['height']}) "
              f"→ resized to {width}x{height}")

    def read(self):
        import numpy as _np
        shot  = self._mss.grab(self._mon)
        frame = _np.array(shot, dtype=_np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_LINEAR)
        return True, frame

    def release(self):
        self._mss.close()


def make_capture(source: str, camera: int = 0, monitor: int = 1,
                 width: int = 1280, height: int = 720):
    """Return a CameraCapture or ScreenCapture depending on `source`."""
    if source == 'screen':
        return ScreenCapture(monitor=monitor, width=width, height=height)
    return CameraCapture(index=camera, width=width, height=height)
