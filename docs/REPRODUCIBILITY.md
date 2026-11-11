# Reproduksi

## Lingkungan yang dipakai

Lihat `logs/environment.json`: Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2, GPU Tesla T4 (Google Colab gratis). Seed 42 (`keras.utils.set_random_seed(seed + fold)`). Hasil GPU dengan mixed precision tidak dijamin identik bit demi bit antar-perangkat; selisih kecil pada akurasi terkorupsi masih mungkin.

## Google Colab

| Sel | Fungsi |
|---|---|
| Sel 1 | Mount Drive → cari/unduh dataset → cache 224×224 → fold → jalankan supervisor latar belakang → tampilkan log → di akhir sesi rekap + ZIP + unduh otomatis |
| Sel 2 | Mandiri: rekap, uji statistik, grafik, ZIP, unduh. Aman dijalankan setelah restart |
| Sel 3 | `HENTIKAN = True` → kirim sinyal berhenti; training berhenti di akhir epoch dengan checkpoint tersimpan |

Mekanisme anti-putus:

- **Supervisor terlepas** (`start_new_session=True`) menjalankan proses *worker*. Notebook hanya membaca log.
- Kode keluar worker: `0` selesai, `2` anggaran sesi habis, `3` daur ulang RAM (RSS > 6 GB atau RAM tersedia < 2,5 GB). Worker yang crash di-restart maksimal 3× tanpa kemajuan.
- **Checkpoint atomik** (`tmp` + `os.replace`): bobot terbaik disimpan di RAM, cadangan ke Drive tiap 3 epoch beserta `state.json` (epoch, LR, best, waktu).
- `results/results.csv` ditambah satu baris per job selesai; job yang sudah ada di file dilewati.
- Anggaran 105 menit per sesi; sisa waktu dipakai untuk rekap dan ZIP.

Struktur folder Drive (`MyDrive/BEEI_Batik/`):

```
data/ cache/ results/ preds/ ckpt/ backup/ figures/ logs/  + BEEI_Batik_hasil_<STATUS>_<waktu>.zip
```

## Lokal / server

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PROJECT_DIR=$PWD/BEEI_Batik
mkdir -p $PROJECT_DIR/data && cp /path/ke/BatikNitik960.zip $PROJECT_DIR/data/
python src/train_pipeline.py          # proses induk -> supervisor -> worker; aman dihentikan & dijalankan ulang
python src/analysis.py                # rekap + statistik + grafik + ZIP di $PROJECT_DIR
```

Variabel lingkungan:

| Variabel | Default | Arti |
|---|---|---|
| `PROJECT_DIR` | `./BEEI_Batik` (lokal) | folder kerja (data, cache, hasil) |
| `BEEI_SMOKE` | kosong | `1` = uji cepat (2 epoch, hanya fold 0 dan EfficientNetB0 untuk group/fs1/fs8) |
| `BEEI_BUDGET_MIN` | `105` | batas waktu training per sesi (menit) |
| `PRETRAINED` | `imagenet` | `none` = tanpa bobot ImageNet (hanya untuk uji pipeline) |

Untuk menghentikan dengan aman di lokal: `PROJECT_DIR=... python src/stop_training.py` setelah mengubah `HENTIKAN = True`, atau buat file `$PROJECT_DIR/logs/BEEI_STOP`.

## Verifikasi yang sudah dilakukan

`src/analysis.py` dijalankan ulang pada `results/results.csv` di repositori ini. Enam tabel (`summary_all`, `table_main_group`, `table_fewshot`, `table_leakage_inflation`, `table_robustness_by_strategy`, `table_robustness_full`) identik dengan file asli, dan `fig_fewshot_curve.png` identik byte demi byte.

`stats_pairwise.csv`, `stats_fewshot_mcnemar.csv`, dan `confusion_best.csv` hanya bisa dibangkitkan ulang bila `predictions/*.npz` tersedia.

## Waktu komputasi

Training tahap 1 per job: S1 21–43 s, S2 149–228 s, S3 164–623 s. Tahap 2: S1 28–64 s, S3 150–318 s. Ditambah waktu evaluasi tujuh korupsi per job tahap 2.

Uji jalan skrip repositori (`BEEI_SMOKE=1 PRETRAINED=none`, CPU, dataset sintetis 960 citra dengan pola nama file asli): 7 job selesai melalui supervisor → worker, termasuk satu daur ulang RAM otomatis. Rekap dan ZIP terbentuk, dan file prediksi berisi semua kunci (`pred_<korupsi>`). Kebocoran protokol random pada uji ini (87,1 / 90,8 / 93,8 / 94,6 %) sama dengan eksperimen asli, karena fold hanya bergantung pada nama file dan seed.
