"""Gradio uygulamasi: fotograftaki cok haneli sayiyi MNIST-CNN ile okur.
Calistirma:  python app.py   ->  http://127.0.0.1:7860"""
import cv2
import numpy as np
import gradio as gr
try:  # iPhone .heic/.heif fotograflari icin
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("UYARI: HEIC destegi icin: pip install pillow-heif")
from predictor import predict_number, annotate, get_model

get_model()


def _load_bgr(path):
    """Goruntuyu sirayla farkli yontemlerle acar: PIL(+heif) -> pillow_heif -> OpenCV -> macOS sips."""
    import os, subprocess, tempfile, traceback
    from PIL import Image, ImageOps
    errors = []

    def finish(im):
        im = ImageOps.exif_transpose(im).convert("RGB")  # telefon yonlendirmesi
        im.thumbnail((2000, 2000))
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    try:
        return finish(Image.open(path))
    except Exception as e:
        errors.append(f"PIL: {e}")
    try:
        import pillow_heif
        hf = pillow_heif.read_heif(path)
        return finish(Image.frombytes(hf.mode, hf.size, hf.data, "raw", hf.mode, hf.stride))
    except Exception as e:
        errors.append(f"pillow_heif: {e}")
    try:
        im = cv2.imread(path)
        if im is not None:
            return im
        errors.append("OpenCV: okuyamadi")
    except Exception as e:
        errors.append(f"OpenCV: {e}")
    try:  # macOS yerlesik donusturucu
        out = os.path.join(tempfile.mkdtemp(), "conv.jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", path, "--out", out],
                       check=True, capture_output=True, timeout=30)
        return finish(Image.open(out))
    except Exception as e:
        errors.append(f"sips: {e}")
    with open(path, "rb") as f:
        head = f.read(32)
    print("GORUNTU ACILAMADI:", path, "| ilk baytlar:", head, "|", errors)
    raise RuntimeError("Hicbir yontem acamadi. Ilk baytlar: %r | %s" % (head, " ; ".join(errors)))


def read_number(path):
    """path: yuklenen goruntunun dosya yolu. Donus: kutulu gorsel, okunan sayi, tablo, sure."""
    if path is None:
        return None, "", [], ""
    path = getattr(path, "name", path)
    try:
        bgr = _load_bgr(path)
    except Exception as e:
        return None, f"Goruntu acilamadi: {e}", [], ""
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = predict_number(bgr)
    if not res["number"]:
        return image, "Rakam bulunamadi", [], ""
    vis = cv2.cvtColor(annotate(bgr, res), cv2.COLOR_BGR2RGB)
    rows = [[i + 1, d, f"{c*100:.1f}%"] for i, (d, c) in enumerate(zip(res["digits"], res["conf"]))]
    return vis, res["number"], rows, f"{res['ms']:.1f} ms ({len(res['digits'])} rakam)"


def read_sketch(data):
    """Tek rakam cizimi. Arka plan rengi koseden tahmin edilir; ondan farkli pikseller murekkeptir."""
    if data is None:
        return {}
    img = data["composite"] if isinstance(data, dict) else data
    if img is None:
        return {}
    img = np.array(img)
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32) * a[..., None] + 255.0 * (1 - a[..., None])  # beyaz uzerine
    elif img.ndim == 3:
        rgb = img[..., :3].astype(np.float32)
    else:
        rgb = np.stack([img] * 3, -1).astype(np.float32)
    gray = rgb.mean(-1)
    bg = np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]]))
    ink = np.abs(gray - bg) > 40
    if ink.sum() < 20:
        return {}
    ys, xs = np.where(ink)
    from digit_segmentation import _to_mnist
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    x = _to_mnist(crop)[None, ..., None]
    p = get_model()(x, training=False).numpy()[0]
    return {str(i): float(p[i]) for i in range(10)}


with gr.Blocks(title="MNIST CNN - Cok Haneli Sayi Okuyucu") as demo:
    gr.Markdown("# MNIST CNN ile Rakam Tanima\nBir sayi fotografi yukleyin; rakamlar ayrilip tek tek siniflandirilir.")
    with gr.Tab("Fotograf yukle"):
        with gr.Row():
            inp = gr.File(label="Sayi fotografi (jpg, png, heic...)", type="filepath",
                          file_types=[".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp"])
            out_img = gr.Image(label="Tespit edilen rakamlar")
        out_num = gr.Textbox(label="Okunan sayi", scale=1)
        out_time = gr.Textbox(label="Tahmin suresi")
        out_tbl = gr.Dataframe(headers=["Sira", "Tahmin", "Guven"], label="Rakam bazinda sonuc")
        btn = gr.Button("Oku", variant="primary")
        btn.click(read_number, inp, [out_img, out_num, out_tbl, out_time])
        inp.change(read_number, inp, [out_img, out_num, out_tbl, out_time])
    with gr.Tab("Tek rakam ciz"):
        sk = gr.Sketchpad(label="Bir rakam cizin", type="numpy", brush=gr.Brush(default_size=14, colors=["#000000"]))
        lab = gr.Label(num_top_classes=3, label="Tahmin")
        sk.change(read_sketch, sk, lab)

if __name__ == "__main__":
    demo.launch()