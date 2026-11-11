# Prediksi out-of-fold

Folder ini berisi satu file `.npz` per job (138 file), nama: `<protokol>_<backbone>_f<fold>_s<seed>_<strategi>.npz`.

| Kunci | Bentuk | Isi |
|---|---|---|
| `test_idx` | (n_test,) | indeks citra uji (urutan baris `splits/groups.csv`) |
| `y_true` | (n_test,) | label benar |
| `y_pred` | (n_test,) | prediksi pada citra bersih |
| `probs` | (n_test, 60) float16 | probabilitas softmax |
| `pred_<korupsi>` | (n_test,) | prediksi pada citra terkorupsi (`fs1`, `fs4`, `fs8` saja) |

Contoh:

```python
import numpy as np, glob
for f in sorted(glob.glob('predictions/fs8_ResNet50_f*_s42_S3_full.npz')):
    d = np.load(f)
    print(f, (d['y_pred'] == d['y_true']).mean(), (d['pred_zoom'] == d['y_true']).mean())
```

File ini dibutuhkan untuk membangkitkan ulang `results/stats_*.csv` dan `results/confusion_best.csv`
(salin ke `$PROJECT_DIR/preds/` lalu jalankan `src/analysis.py`).

> **Untuk pemilik repositori:** salin isi `MyDrive/BEEI_Batik/preds/*.npz` dari Google Drive ke folder ini sebelum push.
> Ukuran total kecil (probabilitas float16, terkompresi), aman untuk GitHub.
