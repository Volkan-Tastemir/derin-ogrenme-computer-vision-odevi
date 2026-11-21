"""Telefonla cekilmis cok haneli sayi fotograflarini tek tek rakamlara ayirir
ve her rakami MNIST formatina (28x28, siyah zemin, beyaz rakam, agirlik merkezine
gore ortali) donusturur."""
import cv2
import numpy as np


def _binarize(gray):
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Koyu murekkep rakamlari silen buyuk bir kapama (closing) ile arka plan isigini tahmin et
    k = max(15, (min(gray.shape) // 3) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(gray, bg, scale=255)
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _to_mnist(crop):
    """crop: ikili (rakam=255) kirpinti -> 28x28 float32 [0,1] (MNIST tarzi)."""
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.uint8)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    # Agirlik merkezini goruntu merkezine tasi (MNIST on isleme adimi)
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        M = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(canvas, M, (28, 28))
    return canvas.astype("float32") / 255.0


def _split_wide(bw, x, w, n):
    """Birlesik (birbirine degen) rakamlari n parcaya, dikey izdusum
    minimumlarindan keserek boler. Donus: [(x_baslangic, genislik), ...]"""
    proj = (bw[:, x:x + w] > 0).sum(axis=0).astype(float)
    proj = np.convolve(proj, np.ones(3) / 3, mode="same")
    cuts = [0]
    for k in range(1, n):
        ideal = int(round(k * w / n))
        lo, hi = max(cuts[-1] + 3, ideal - int(0.3 * w / n)), min(w - 3, ideal + int(0.3 * w / n))
        c = lo + int(np.argmin(proj[lo:hi])) if hi > lo else ideal
        cuts.append(c)
    cuts.append(w)
    return [(x + a, b - a) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 2]


def segment_digits(bgr):
    """Donus: (digits[N,28,28], boxes[(x,y,w,h)], binary_image)"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bw = _binarize(gray)
    # Rakamlarin ustunden gecen ince acik cizgileri (tarama/isik cizgisi) kapat
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = [tuple(stats[i][:4]) + (stats[i][4],) for i in range(1, n)]  # x,y,w,h,area
    if not comps:
        return np.zeros((0, 28, 28), np.float32), [], bw
    H = bw.shape[0]
    # Rakam adaylari: yeterince uzun bilesenler (gurultu, catlak, cizgileri ele)
    hmax = max(c[3] for c in comps if c[3] < 0.98 * H) if any(c[3] < 0.98 * H for c in comps) else H
    cand = [c for c in comps if c[3] >= 0.55 * hmax and c[3] < 0.98 * H and c[4] > 0.04 * hmax * hmax]
    if not cand:
        return np.zeros((0, 28, 28), np.float32), [], bw
    # Ana metin seridi: en yuksek bilesenin dikey araligiyla ortusenler
    cy = np.median([c[1] + c[3] / 2 for c in cand])
    cand = [c for c in cand if abs((c[1] + c[3] / 2) - cy) < 0.5 * hmax]
    cand.sort(key=lambda c: c[0])

    # Tek rakam genisligi tahmini: "1" haric medyan genislik
    widths = np.array([c[2] for c in cand], float)
    single = np.median(widths[widths < 1.35 * np.median(widths)])
    boxes = []
    for (x, y, w, h, a) in cand:
        n_d = int(round(w / single)) if w > 1.4 * single else 1
        if n_d > 1:
            for (sx, sw) in _split_wide(bw, x, w, n_d):
                boxes.append((sx, y, sw, h))
        else:
            boxes.append((x, y, w, h))

    digits = []
    final = []
    for (x, y, w, h) in boxes:
        crop = bw[y:y + h, x:x + w]
        # kirpintidaki diger rakam parcalarini (komsu) temizlemek icin sikica kirp
        ys, xs = np.where(crop > 0)
        if len(xs) == 0:
            continue
        crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        digits.append(_to_mnist(crop))
        final.append((x + int(xs.min()), y + int(ys.min()), crop.shape[1], crop.shape[0]))
    return np.stack(digits).astype("float32"), final, bw
