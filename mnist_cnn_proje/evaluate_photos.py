"""5 telefon fotografini modele ver, tahmin ile gercegi karsilastir, gorselleri kaydet."""
import json, cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predictor import predict_number, annotate

PHOTOS = {  # dosya -> gercek sayi (fotograftan elle okundu)
    "IMG_4819.jpg": "30875590002",
    "IMG_4820.jpg": "30875590003",
    "IMG_4822.jpg": "5840578867",
    "IMG_4823.jpg": "7330038818666139",
    "IMG_4824.jpg": "6320431",
}
UP = "/mnt/user-data/uploads/"
results = []
for name, truth in PHOTOS.items():
    bgr = cv2.imread(UP + name)
    res = predict_number(bgr)
    pred = res["number"]
    n = min(len(pred), len(truth))
    correct = sum(a == b for a, b in zip(pred, truth)) if len(pred) == len(truth) else None
    wrong = [(i + 1, truth[i], pred[i], round(res["conf"][i], 3))
             for i in range(n) if pred[i] != truth[i]] if len(pred) == len(truth) else []
    results.append(dict(file=name, truth=truth, pred=pred, n_digits=len(truth),
                        n_segmented=len(pred), digit_correct=correct,
                        exact=pred == truth, ms=res["ms"],
                        min_conf=min(res["conf"]) if res["conf"] else 0, wrong=wrong))
    # Gorsel: kutulu fotograf + 28x28 kirpintilar
    vis = cv2.cvtColor(annotate(bgr, res), cv2.COLOR_BGR2RGB)
    k = len(res["crops"])
    fig = plt.figure(figsize=(max(8, k * 0.75), 3.6))
    ax = fig.add_axes([0, 0.45, 1, 0.55]); ax.imshow(vis); ax.axis("off")
    ax.set_title(f"{name}  |  Gercek: {truth}  |  Tahmin: {pred}", fontsize=10,
                 color="green" if pred == truth else "red")
    for i in range(k):
        a = fig.add_axes([i / k + 0.003, 0.02, 1 / k - 0.006, 0.36])
        a.imshow(res["crops"][i], cmap="gray"); a.axis("off")
        ok = len(pred) == len(truth) and pred[i] == truth[i]
        a.set_title(f"{pred[i]} ({res['conf'][i]*100:.0f}%)", fontsize=8,
                    color="green" if ok else "red")
    fig.savefig(f"result_{name[:-4]}.png", dpi=130); plt.close(fig)

tot = sum(r["n_digits"] for r in results)
cor = sum(r["digit_correct"] or 0 for r in results)
summary = dict(photos=results, total_digits=tot, correct_digits=cor,
               digit_accuracy=cor / tot, exact_photos=sum(r["exact"] for r in results))
json.dump(summary, open("photo_results.json", "w"), indent=2, ensure_ascii=False)
for r in results:
    print(r["file"], r["truth"], r["pred"], "OK" if r["exact"] else "HATA", r["wrong"], f"{r['ms']:.1f}ms")
print("Rakam dogrulugu:", cor, "/", tot, f"= {cor/tot:.3%}")
