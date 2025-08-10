import time
from typing import List, Tuple

import cv2
import mss
import numpy as np


class DetectedElement:
    def __init__(self, x: int, y: int, w: int, h: int, kind: str):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.kind = kind

    @property
    def center_y(self) -> int:
        return self.y + self.h // 2


def capture_primary_screen() -> Tuple[np.ndarray, Tuple[int, int]]:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return frame_bgr, (monitor["width"], monitor["height"])


def detect_ui_elements(image_bgr: np.ndarray) -> List[DetectedElement]:
    original_h, original_w = image_bgr.shape[:2]
    scale = 0.6
    resized = cv2.resize(image_bgr, (int(original_w * scale), int(original_h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elements: List[DetectedElement] = []
    min_area = int((original_w * original_h) * 0.00003)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        w_o = int(w / scale)
        h_o = int(h / scale)
        x_o = int(x / scale)
        y_o = int(y / scale)

        if w_o < 40 or h_o < 18:
            continue

        extent = float(area) / float(w * h + 1e-6)
        if extent < 0.35:
            continue

        aspect = w_o / float(h_o)
        x1_r = max(0, x)
        y1_r = max(0, y)
        x2_r = min(resized.shape[1], x + w)
        y2_r = min(resized.shape[0], y + h)
        roi = gray[y1_r:y2_r, x1_r:x2_r]
        mean_intensity = float(np.mean(roi)) if roi.size > 0 else 0.0

        kind = "component"
        if aspect >= 4.0 and 18 <= h_o <= 80 and mean_intensity >= 160:
            kind = "input"
        elif 1.2 <= aspect <= 4.0 and 25 <= h_o <= 120 and extent >= 0.45:
            kind = "button"

        elements.append(DetectedElement(x_o, y_o, w_o, h_o, kind))

    return elements


def draw_elements_progressive(base_bgr: np.ndarray, elements: List[DetectedElement], scan_y: int) -> np.ndarray:
    overlay = base_bgr.copy()
    for el in elements:
        if el.center_y <= scan_y:
            color = (0, 0, 255)
            if el.kind == "button":
                color = (255, 0, 0)
            elif el.kind == "input":
                color = (0, 200, 0)
            cv2.rectangle(overlay, (el.x, el.y), (el.x + el.w, el.y + el.h), color, 2)
            label = el.kind
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = max(el.x, 0)
            ty = max(el.y - 6, th + 4)
            cv2.rectangle(overlay, (tx, ty - th - 4), (tx + tw + 6, ty + 2), (0, 0, 0), -1)
            cv2.putText(overlay, label, (tx + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    bar_y = max(0, min(scan_y, base_bgr.shape[0] - 1))
    cv2.rectangle(overlay, (0, bar_y - 3), (base_bgr.shape[1], bar_y + 3), (0, 255, 255), -1)
    return overlay


def run():
    window_name = "Scan UI"

    base_frame, (w, h) = capture_primary_screen()
    elements: List[DetectedElement] = detect_ui_elements(base_frame)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    step = 14
    for scan_y in range(0, h, step):
        output = draw_elements_progressive(base_frame, elements, scan_y)
        cv2.imshow(window_name, output)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            return

    final = draw_elements_progressive(base_frame, elements, h)
    cv2.imshow(window_name, final)
    while True:
        if cv2.waitKey(50) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()


