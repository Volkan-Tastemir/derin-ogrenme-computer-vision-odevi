"""Model yukleme + cok haneli sayi tahmini (hem uygulama hem test betigi kullanir)."""
import time
import cv2
import numpy as np
from tensorflow import keras
from digit_segmentation import segment_digits

_model = None


def get_model(path="mnist_cnn.keras"):
    global _model
    if _model is None:
        _model = keras.models.load_model(path)
        _model.predict(np.zeros((1, 28, 28, 1), np.float32), verbose=0)  # isinma
    return _model


def predict_number(bgr):
    """bgr: OpenCV goruntusu. Donus: dict(sayi, rakamlar, guvenler, kutular, sure_ms, crops)"""
    model = get_model()
    t0 = time.perf_counter()
    digits, boxes, _ = segment_digits(bgr)
    if len(digits) == 0:
        return dict(number="", digits=[], conf=[], boxes=[], ms=0.0, crops=digits)
    proba = model(digits[..., None], training=False).numpy()
    ms = (time.perf_counter() - t0) * 1000
    pred = proba.argmax(1)
    return dict(number="".join(map(str, pred)), digits=pred.tolist(),
                conf=proba.max(1).tolist(), boxes=boxes, ms=ms, crops=digits, proba=proba)


def annotate(bgr, res):
    vis = bgr.copy()
    th = max(2, bgr.shape[1] // 450)
    for (x, y, w, h), d, c in zip(res["boxes"], res["digits"], res["conf"]):
        col = (0, 160, 0) if c >= 0.9 else (0, 140, 255) if c >= 0.6 else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, th)
        fs = max(0.8, bgr.shape[1] / 1100)
        cv2.putText(vis, str(d), (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th + 1)
    return vis
