import os, sys
from PIL import Image, ImageOps
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("UYARI: HEIC icin 'pip install pillow-heif' gerekir")

src = sys.argv[1] if len(sys.argv) > 1 else "photos"
dst = sys.argv[2] if len(sys.argv) > 2 else "photos_clean"
os.makedirs(dst, exist_ok=True)
EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")
for name in sorted(os.listdir(src)):
    if not name.lower().endswith(EXT):
        continue
    im = Image.open(os.path.join(src, name))
    im = ImageOps.exif_transpose(im).convert("RGB")
    clean = Image.new("RGB", im.size); clean.paste(im)
    out = os.path.join(dst, os.path.splitext(name)[0] + ".jpg")
    clean.save(out, "JPEG", quality=95)
    left = Image.open(out).getexif()
    print(f"{name} -> {out}  (kalan EXIF etiketi: {len(left)})")
