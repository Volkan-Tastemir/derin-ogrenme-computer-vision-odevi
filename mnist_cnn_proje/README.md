# MNIST CNN - Cok Haneli Sayi Okuyucu

1. MNIST dosyalarini `../data/` klasorune koyun (train/t10k idx3/idx1 .gz) - http://yann.lecun.com/exdb/mnist/
2. `pip install -r requirements.txt`
3. Egitim:            `python train.py`            (mnist_cnn.keras, metrics.json, grafikler uretir)
4. Foto testi:        `python evaluate_photos.py`  (fotograf yollarini PHOTOS/UP ile duzenleyin)
5. Uygulama (Gradio): `python app.py`              -> http://127.0.0.1:7860

Dosyalar: train.py (egitim) - digit_segmentation.py (rakam ayirma + MNIST bicimi) - predictor.py (tahmin) - app.py (arayuz)
