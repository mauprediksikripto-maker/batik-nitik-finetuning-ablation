# Hasil eksperimen

File ini dibangkitkan otomatis oleh `scripts/make_results_md.py` dari `results/*.csv`. Jangan diedit manual.

## Tahap 1 — protokol group (bebas kebocoran), data penuh

| backbone | strategy | Accuracy (%) | Macro-F1 (%) | Trainable params | Train time (s) |
|---|---|---|---|---|---|
| EfficientNetB0 | S0_scratch | 1.67 ± 0.00 | 0.05 ± 0.00 | 4,084,408 | 271 |
| EfficientNetB0 | S1_frozen | 100.00 ± 0.00 | 100.00 ± 0.00 | 76,860 | 43 |
| EfficientNetB0 | S2_partial | 100.00 ± 0.00 | 100.00 ± 0.00 | 2,726,140 | 228 |
| EfficientNetB0 | S3_full | 100.00 ± 0.00 | 100.00 ± 0.00 | 4,042,392 | 239 |
| MobileNetV2 | S1_frozen | 100.00 ± 0.00 | 100.00 ± 0.00 | 76,860 | 21 |
| MobileNetV2 | S2_partial | 100.00 ± 0.00 | 100.00 ± 0.00 | 1,740,220 | 149 |
| MobileNetV2 | S3_full | 100.00 ± 0.00 | 100.00 ± 0.00 | 2,266,620 | 164 |
| ResNet50 | S0_scratch | 51.87 ± 63.93 | 49.18 ± 66.87 | 23,657,532 | 712 |
| ResNet50 | S1_frozen | 100.00 ± 0.00 | 100.00 ± 0.00 | 122,940 | 24 |
| ResNet50 | S2_partial | 100.00 ± 0.00 | 100.00 ± 0.00 | 16,192,060 | 216 |
| ResNet50 | S3_full | 100.00 ± 0.00 | 100.00 ± 0.00 | 23,604,412 | 623 |

ResNet50 S0 hanya 2 fold (per fold: 97.1 / 6.7 %). MobileNetV2 S0 tidak dijalankan (dikeluarkan dari rencana, batas waktu GPU).

**Inflasi kebocoran (random − group):** 0.0–0.0 pp untuk semua backbone × S1/S3; kebocoran rotasi pada protokol random rata-rata 91.6 % (rentang 87.1–94.6 %).

> Catatan: kolom `latency_ms` di `table_main_group.csv` (tahap 1) diukur dengan pemanggilan eager dan **tidak valid**; latensi yang valid (median `tf.function`) ada di `table_fewshot.csv`.

## Tahap 2 — data sedikit + ketahanan korupsi (rata-rata 3 backbone × 4 fold)

| Citra/kelas | Strategi | Bersih (%) | Rata2 korupsi (%) | blur | noise | jpeg | dark | lowcontrast | zoom | rot15 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S1_frozen | 99.9 | 73.3 | 45.1 | 85.3 | 80.0 | 98.4 | 96.5 | 39.4 | 68.6 |
| 1 | S3_full | 99.8 | 73.9 | 48.8 | 84.4 | 79.5 | 96.8 | 96.3 | 41.5 | 69.9 |
| 4 | S1_frozen | 99.9 | 77.2 | 50.3 | 88.0 | 85.3 | 99.5 | 99.1 | 43.5 | 74.4 |
| 4 | S3_full | 100.0 | 78.7 | 53.4 | 87.7 | 86.3 | 98.6 | 98.8 | 47.5 | 78.6 |
| 8 | S1_frozen | 100.0 | 78.8 | 53.9 | 89.1 | 86.3 | 99.8 | 99.4 | 44.9 | 78.0 |
| 8 | S3_full | 100.0 | 81.4 | 57.4 | 88.3 | 88.0 | 99.8 | 99.5 | 52.9 | 83.8 |

### Selisih ketahanan per backbone (rata-rata 7 korupsi)

| Citra/kelas | Backbone | S1 korupsi (%) | S3 korupsi (%) | S3 − S1 (pp) |
|---|---|---|---|---|
| 1 | EfficientNetB0 | 82.7 | 81.8 | -0.9 |
| 1 | MobileNetV2 | 65.1 | 66.3 | 1.2 |
| 1 | ResNet50 | 72.2 | 73.6 | 1.4 |
| 4 | EfficientNetB0 | 86.0 | 86.2 | 0.3 |
| 4 | MobileNetV2 | 69.6 | 71.5 | 1.9 |
| 4 | ResNet50 | 75.9 | 78.4 | 2.5 |
| 8 | EfficientNetB0 | 87.1 | 88.3 | 1.2 |
| 8 | MobileNetV2 | 71.3 | 71.7 | 0.4 |
| 8 | ResNet50 | 77.9 | 84.1 | 6.2 |

### Biaya

| Strategi | Waktu latih min (s) | Waktu latih maks (s) |
|---|---|---|
| S1_frozen | 28 | 64 |
| S3_full | 150 | 318 |

| Backbone | Latensi min (ms) | Latensi maks (ms) |
|---|---|---|
| EfficientNetB0 | 9.0 | 11.3 |
| MobileNetV2 | 6.5 | 8.8 |
| ResNet50 | 10.4 | 15.0 |

Latensi = median inferensi satu citra (`tf.function`, Tesla T4, mixed precision).

## Uji statistik (McNemar eksak, prediksi out-of-fold gabungan 960 citra, koreksi Holm)

Kondisi bersih: tidak ada perbedaan S1 vs S3 yang signifikan (p_holm minimum = 0.375).

Kondisi terkorupsi:

| Citra/kelas | Uji signifikan (Holm p<0,05) | S1 menang | S3 menang |
|---|---|---|---|
| 1 | 11 | 6 | 5 |
| 4 | 10 | 4 | 6 |
| 8 | 10 | 2 | 8 |
