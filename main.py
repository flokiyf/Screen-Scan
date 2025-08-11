import time
import os
from typing import List, Tuple
import argparse

import cv2
import mss
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # onnxruntime optional
    ort = None  # type: ignore


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


def letterbox(image: np.ndarray, new_shape: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, (left, top)


def load_yolo_session(model_path: str) -> Tuple[object, int]:
    if ort is None:
        return None, 0
    if not os.path.isfile(model_path):
        return None, 0
    providers = [
        "CPUExecutionProvider",
    ]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        input_h = session.get_inputs()[0].shape[-1] or 640
        return session, int(input_h)
    except Exception:
        return None, 0


def nms_boxes(boxes: List[List[float]], scores: List[float], iou_thres: float) -> List[int]:
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(
        bboxes=[list(map(int, b)) for b in boxes],
        scores=scores,
        score_threshold=0.0,
        nms_threshold=iou_thres,
    )
    if len(idxs) == 0:
        return []
    return [int(i) for i in (idxs.flatten() if hasattr(idxs, "flatten") else idxs)]


def detect_with_yolo(image_bgr: np.ndarray, session: object, input_size: int, conf_thres: float = 0.30, iou_thres: float = 0.45) -> List[DetectedElement]:
    try:
        img_in, scale, (pad_x, pad_y) = letterbox(image_bgr, input_size)
        img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        tensor = img_rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        pred = outputs[0]

        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.shape[0] in (6, 84, 85):
            pred = pred.transpose(1, 0)

        num_cols = pred.shape[1]
        if num_cols < 6:
            return []

        boxes_xywh = pred[:, 0:4]
        has_obj = num_cols > 5
        if has_obj:
            obj = pred[:, 4]
            cls_scores = pred[:, 5:]
        else:
            obj = np.ones((pred.shape[0],), dtype=pred.dtype)
            cls_scores = pred[:, 4:]

        cls_idx = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_idx]
        conf = obj * cls_conf

        mask = conf >= conf_thres
        boxes_xywh = boxes_xywh[mask]
        conf = conf[mask]
        cls_idx = cls_idx[mask]

        if boxes_xywh.size == 0:
            return []

        cx, cy, bw, bh = boxes_xywh.T
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        h, w = image_bgr.shape[:2]
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(float).tolist()
        scores = conf.tolist()

        keep = nms_boxes(boxes_xyxy, scores, iou_thres)
        kept = []
        for i in keep:
            x1i, y1i, x2i, y2i = boxes_xyxy[i]
            w_i = max(1, int(x2i - x1i))
            h_i = max(1, int(y2i - y1i))
            x_i = int(x1i)
            y_i = int(y1i)
            label = "component"
            if cls_idx[i] == 0:
                label = "button"
            elif cls_idx[i] == 1:
                label = "input"
            kept.append(DetectedElement(x_i, y_i, w_i, h_i, label))
        return kept
    except Exception:
        return []


def list_template_files(templates_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(templates_dir):
        return []
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files: List[Tuple[str, str]] = []
    for root, _, filenames in os.walk(templates_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                files.append((os.path.join(root, fname), fname))
    return files


def detect_with_templates(image_bgr: np.ndarray, templates_dir: str, threshold: float = 0.88, scales: List[float] | None = None) -> List[DetectedElement]:
    if scales is None:
        scales = [0.85, 0.95, 1.0, 1.05, 1.15]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    files = list_template_files(templates_dir)
    if not files:
        return []

    boxes: List[List[float]] = []
    scores: List[float] = []
    labels: List[str] = []

    for path, filename in files:
        label = os.path.splitext(filename)[0]
        templ = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if templ is None:
            continue
        th, tw = templ.shape[:2]
        for s in scales:
            sw = max(8, int(tw * s))
            sh = max(8, int(th * s))
            if sw >= gray.shape[1] or sh >= gray.shape[0]:
                continue
            templ_s = cv2.resize(templ, (sw, sh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            res = cv2.matchTemplate(gray, templ_s, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for y, x in zip(*loc):
                score = float(res[y, x])
                boxes.append([float(x), float(y), float(x + sw), float(y + sh)])
                scores.append(score)
                labels.append(label)

    keep = nms_boxes(boxes, scores, 0.30)
    out: List[DetectedElement] = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        w = int(max(1, x2 - x1))
        h = int(max(1, y2 - y1))
        x = int(x1)
        y = int(y1)
        kind = labels[i]
        out.append(DetectedElement(x, y, w, h, kind))
    return out


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
            if el.kind == "button" or ("button" in el.kind.lower() or "postul" in el.kind.lower()):
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


def run(target_filter: set | None = None):
    window_name = "Scan UI"
    model_path = os.path.join("models", "yolov10n-ui.onnx")
    session, input_sz = load_yolo_session(model_path)

    base_frame, (w, h) = capture_primary_screen()
    if session:
        elements = detect_with_yolo(base_frame, session, input_sz or 640)
        if not elements:
            elements = detect_ui_elements(base_frame)
    else:
        elements = detect_ui_elements(base_frame)

    # Template matching augmentation (directory: ./templates)
    templates_dir = ARGS.templates_dir if 'ARGS' in globals() else 'templates'
    template_thres = ARGS.template_thres if 'ARGS' in globals() else 0.88
    template_scales = ARGS.template_scales if 'ARGS' in globals() else [0.85, 0.95, 1.0, 1.05, 1.15]
    template_elements = detect_with_templates(base_frame, templates_dir, template_thres, template_scales)
    if template_elements:
        elements.extend(template_elements)

    if target_filter:
        elements = [e for e in elements if e.kind in target_filter]

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="", help="comma-separated kinds to keep: button,input,component or template labels (e.g., postuler)")
    parser.add_argument("--templates-dir", type=str, default="templates", help="directory containing template images")
    parser.add_argument("--template-thres", dest="template_thres", type=float, default=0.88, help="template match threshold [0-1]")
    parser.add_argument("--template-scales", dest="template_scales", type=str, default="0.85,0.95,1.0,1.05,1.15", help="comma-separated scales for template matching")
    ARGS = parser.parse_args()
    try:
        ARGS.template_scales = [float(s.strip()) for s in ARGS.template_scales.split(',') if s.strip()]
    except Exception:
        ARGS.template_scales = [0.85, 0.95, 1.0, 1.05, 1.15]
    target = set([t.strip() for t in ARGS.target.split(',') if t.strip()]) or None
    run(target_filter=target)


