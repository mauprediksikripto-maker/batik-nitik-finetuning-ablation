# Kamus data

Semua akurasi disimpan sebagai proporsi (0–1) kecuali disebut lain. `pp` = poin persentase.

## results/results.csv — 138 baris, satu baris per job (protokol × backbone × strategi × fold × seed)

| Kolom | Arti |
|---|---|
| `job` | ID unik `protokol\|backbone\|strategi\|f<fold>\|s<seed>` |
| `protocol` | `group`, `random`, `fs1`, `fs4`, `fs8` (lihat PROTOCOL.md) |
| `backbone`, `strategy`, `fold`, `seed` | faktor eksperimen |
| `n_train`, `n_val`, `n_test` | jumlah citra latih/validasi/uji |
| `leakage` | proporsi citra uji yang punya saudara rotasi di data latih |
| `trainable_params`, `total_params` | parameter yang dilatih pada tahap terakhir / total |
| `train_time_s` | waktu latih kumulatif (head + fine-tune), detik |
| `epochs` | epoch kumulatif |
| `wall_min` | waktu dinding job termasuk evaluasi, menit |
| `acc`, `prec_macro`, `rec_macro`, `f1_macro`, `top5` | metrik pada citra uji bersih |
| `latency_ms` | median inferensi satu citra (valid hanya untuk protokol `fs*`) |
| `shots` | citra latih per kelas (1/4/8; `group` = 8; `random` kosong) |
| `acc_<korupsi>` | akurasi pada citra uji terkorupsi (`fs*` saja) |
| `acc_corrupt_mean` | rata-rata 7 akurasi korupsi |

## Tabel turunan

| File | Isi |
|---|---|
| `summary_all.csv` | rata-rata dan SD antar-fold per protokol × backbone × strategi |
| `table_main_group.csv` | Tabel tahap 1 (protokol group), akurasi & F1 sebagai teks `mean ± SD` (%) |
| `table_leakage_inflation.csv` | akurasi group vs random dan selisihnya (`inflation_pp`) |
| `table_fewshot.csv` | tahap 2 per shots × backbone × strategi; `corrupt_mean` = rata-rata korupsi; `robust_ratio` = `corrupt_mean / acc_mean` |
| `table_robustness_full.csv` | akurasi tiap korupsi per shots × backbone × strategi |
| `table_robustness_by_strategy.csv` | idem, dirata-rata atas backbone dan fold (`n_backbone_fold` = 12, matched) |
| `stats_pairwise.csv` | tahap 1: McNemar eksak (+ t-test berpasangan antar-fold) antar strategi per backbone, koreksi Holm |
| `stats_fewshot_mcnemar.csv` | tahap 2: McNemar eksak S1 vs S3 per shots × kondisi (bersih + 7 korupsi) × backbone; `A_only_correct`/`B_only_correct` = citra yang hanya benar oleh A/B |
| `confusion_best.csv` | matriks konfusi 60×60 model terbaik tahap 1 (EfficientNetB0 S1, out-of-fold 960 citra) |
| `manuscript_numbers.json` | ringkasan angka yang dikutip naskah |

## splits/groups.csv

| Kolom | Arti |
|---|---|
| `path` | path relatif citra di arsip Mendeley |
| `label` | indeks kelas 0–59 (urutan string nama kelas, mis. "1 Sekar Kemuning", "10 Sekar Ketongkeng", …) |
| `group` | potongan kain 0–3 (dari nama file) |
| `method` | versi pengelompokan (`filename-v2`) |

## logs/

| File | Isi |
|---|---|
| `config.json` | backbone, strategi, fold, seed, shots, daftar korupsi |
| `environment.json` | versi Python/TF/Keras dan GPU |
| `plan.json` | 132 job yang direncanakan, berurutan menurut prioritas |
| `history/*.history.csv` | riwayat per epoch (`accuracy, loss, val_accuracy, val_loss, learning_rate`) untuk tahap fine-tune/scratch; head S1 dilatih pada fitur dan tidak menulis riwayat |

## predictions/*.npz

Satu file per job: `test_idx`, `y_true`, `y_pred`, `probs` (float16, n_test × 60), dan untuk `fs*` juga `pred_<korupsi>`. Lihat `predictions/README.md`.
