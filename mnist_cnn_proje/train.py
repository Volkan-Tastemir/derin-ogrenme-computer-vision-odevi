"""MNIST rakam siniflandirma - CNN egitimi (TensorFlow / Keras)."""
import gzip, json, time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
tf.keras.utils.set_random_seed(SEED)
DATA = "../data"


def load_images(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)


def load_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


x_all = load_images(f"{DATA}/train-images-idx3-ubyte.gz")
y_all = load_labels(f"{DATA}/train-labels-idx1-ubyte.gz")
x_test = load_images(f"{DATA}/t10k-images-idx3-ubyte.gz")
y_test = load_labels(f"{DATA}/t10k-labels-idx1-ubyte.gz")

# 60.000 egitim -> 54.000 egitim + 6.000 dogrulama (sinif dengesini koruyarak)
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(x_all))
n_val = 6000
val_idx, tr_idx = idx[:n_val], idx[n_val:]
x_train, y_train = x_all[tr_idx], y_all[tr_idx]
x_val, y_val = x_all[val_idx], y_all[val_idx]


def prep(x):
    return (x.astype("float32") / 255.0)[..., None]


x_train, x_val, x_test_p = prep(x_train), prep(x_val), prep(x_test)
print("train", x_train.shape, "val", x_val.shape, "test", x_test_p.shape)

EPOCHS, BATCH, LR = 10, 128, 1e-3


def build_model():
    # Veri artirma sadece egitimde aktif; gercek foto/kamera varyasyonuna dayaniklilik saglar
    aug = keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomZoom(0.12),
    ], name="augmentation")
    return keras.Sequential([
        layers.Input((28, 28, 1)),
        aug,
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ], name="mnist_cnn")


model = build_model()
model.compile(optimizer=keras.optimizers.Adam(LR),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

ckpt = keras.callbacks.ModelCheckpoint("mnist_cnn.keras", monitor="val_accuracy",
                                       save_best_only=True)
t0 = time.time()
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                 epochs=EPOCHS, batch_size=BATCH, callbacks=[ckpt], verbose=2)
train_time = time.time() - t0
print(f"Egitim suresi: {train_time:.1f} sn")

best = keras.models.load_model("mnist_cnn.keras")
test_loss, test_acc = best.evaluate(x_test_p, y_test, verbose=0)
proba = best.predict(x_test_p, batch_size=256, verbose=0)
pred = proba.argmax(1)

p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro")
print(classification_report(y_test, pred, digits=4))

# Tahmin suresi: tek goruntu (batch=1) ve toplu
best.predict(x_test_p[:1], verbose=0)  # isinma
ts = []
for i in range(200):
    t = time.perf_counter()
    best(x_test_p[i:i + 1], training=False)
    ts.append(time.perf_counter() - t)
single_ms = float(np.mean(ts) * 1000)
t = time.perf_counter()
best.predict(x_test_p, batch_size=256, verbose=0)
batch_ms = (time.perf_counter() - t) / len(x_test_p) * 1000

metrics = dict(
    test_accuracy=float(test_acc), test_loss=float(test_loss),
    macro_precision=float(p), macro_recall=float(r), macro_f1=float(f1),
    best_val_accuracy=float(max(hist.history["val_accuracy"])),
    train_time_sec=train_time, inference_ms_single=single_ms,
    inference_ms_per_image_batch=batch_ms,
    params=int(best.count_params()), epochs=EPOCHS, batch=BATCH, lr=LR,
    n_train=len(x_train), n_val=len(x_val), n_test=len(x_test_p),
    tf_version=tf.__version__,
)
json.dump(metrics, open("metrics.json", "w"), indent=2)
json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()},
          open("history.json", "w"))
print(json.dumps(metrics, indent=2))

# Grafikler
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.close()

cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.imshow(cm, cmap="Blues")
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("Tahmin"); ax.set_ylabel("Gercek"); ax.set_title("Test Confusion Matrix")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()
