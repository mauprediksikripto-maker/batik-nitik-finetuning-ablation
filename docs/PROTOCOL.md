# Protokol eksperimen

Semua definisi di bawah diambil langsung dari `src/train_pipeline.py`.

## 1. Data dan grup

- Batik Nitik 960: 60 kelas × 4 potongan kain × 4 rotasi = 960 citra, di-resize sekali ke 224×224 (cache `uint8`).
- **Grup = nomor potongan kain**, dibaca dari nama file dengan regex
  `^\s*\d+\s+.+?\s(\d+)(?:_rotate_\d+)?\.[A-Za-z]+$` (versi pengelompokan `filename-v2`).
- Hasil pengelompokan tersimpan di `splits/groups.csv` (`path, label, group, method`), 16 citra per kelas, 4 per grup.

## 2. Protokol evaluasi (4 fold, seed 42)

| Protokol | Uji | Validasi | Latih | Kebocoran rotasi |
|---|---|---|---|---|
| `group` (bebas bocor) | potongan *k* | potongan *k*+1 (mod 4) | 2 potongan lain (480 citra) | 0 % |
| `random` (pembanding) | StratifiedKFold split *k* | split *k*+1 | 2 split lain | 91,6 % rata-rata (87,1–94,6 %) |
| `fs1` | sama dengan `group` | sama | 1 citra/kelas: rotasi 0° dari potongan latih pertama (60 citra) | 0 % |
| `fs4` | sama | sama | 4 citra/kelas: semua rotasi potongan latih pertama (240 citra) | 0 % |
| `fs8` | sama | sama | 8 citra/kelas: kedua potongan latih (480 citra) | 0 % |

Kebocoran = proporsi citra uji yang memiliki "saudara rotasi" (kelas dan potongan sama) di data latih. Setiap fold diperiksa tidak beririsan (`assert`).

## 3. Strategi

| Kode | Backbone | Pelatihan | LR | Epoch maks / patience | Augmentasi |
|---|---|---|---|---|---|
| S0 scratch | dilatih dari nol | seluruh jaringan | 1e-3 | 30 / 10 | ya |
| S1 frozen | ImageNet, beku | head (Dropout 0,3 + Dense softmax) dilatih pada fitur GAP yang di-cache | 1e-3 | 60 / 8 | **tidak** |
| S2 partial | ImageNet | 25 % layer teratas dibuka (head dilatih dulu) | 1e-4 | 15 / 4 | ya |
| S3 full | ImageNet | seluruh backbone dibuka (head dilatih dulu) | 3e-5 | 15 / 4 | ya |

- BatchNorm tetap beku; backbone dipanggil dengan `base(x, training=False)`.
- Augmentasi: `RandomFlip('horizontal_and_vertical')`, `RandomZoom(0.1)`, `RandomTranslation(0.1, 0.1)`, `RandomContrast(0.1)`.
- Optimizer Adam, batch 32, mixed precision float16. Early stopping pada `val_loss` (bobot terbaik dipulihkan); untuk S0/S2/S3 juga `ReduceLROnPlateau` (faktor 0,2, patience = ½ patience early stopping).
- S2 dan S3 memakai head hasil S1 pada fold yang sama sebagai *warm-up*, lalu membuka layer backbone.
- **Perancu:** S1 tanpa augmentasi vs S3 dengan augmentasi → yang dibandingkan adalah dua resep.

## 4. Korupsi uji (satu tingkat keparahan, deterministik)

| Nama | Operasi (PIL / NumPy) |
|---|---|
| `blur` | GaussianBlur radius 2 |
| `noise` | derau Gaussian σ = 25 (seed 0) |
| `jpeg` | kompresi JPEG kualitas 15 |
| `dark` | Brightness × 0,5 |
| `lowcontrast` | Contrast × 0,5 |
| `zoom` | crop tengah 60 % lalu resize ke 224 |
| `rot15` | rotasi 15° (bilinear) |

Korupsi hanya diterapkan pada citra uji; model tidak pernah melihatnya saat latih.

## 5. Metrik dan statistik

- Akurasi, presisi/recall/F1 makro, top-5, akurasi tiap korupsi dan rata-ratanya (`acc_corrupt_mean`).
- Latensi: median waktu inferensi satu citra memakai `tf.function` (tahap 2 saja; tahap 1 tidak valid).
- **McNemar eksak** pada prediksi out-of-fold gabungan (960 citra) untuk pasangan strategi pada backbone dan kondisi yang sama; koreksi **Holm** per keluarga uji.
- Perbandingan S1 vs S3 selalu *matched*: backbone dan fold yang sama.

## 6. Rencana job

`logs/plan.json` berisi 132 job yang direncanakan (urutan = prioritas). `results/results.csv` berisi 138 baris: 132 job rencana ditambah 6 job S0 dari rencana tahap 1 sebelumnya (EfficientNetB0 4 fold, ResNet50 2 fold).
