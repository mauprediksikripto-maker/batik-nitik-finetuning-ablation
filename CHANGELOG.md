# Changelog

## v1.0.0 — 2026-11-11 (versi naskah BEEI)

Hasil final: 138 job (132 job rencana + 6 job S0 tahap 1), status `LENGKAP`.

### Tahap 2 — data sedikit dan ketahanan
- Ditambahkan protokol `fs1`, `fs4`, `fs8` (1/4/8 citra latih per kelas) di atas fold `group`.
- Ditambahkan 7 korupsi uji (blur, noise, JPEG, gelap, kontras rendah, zoom, rotasi 15°).
- Strategi tahap 2 dibatasi pada S1 dan S3; S2 serta sisa S0 dikeluarkan dari rencana karena batas waktu GPU gratis.
- Uji McNemar eksak S1 vs S3 per kondisi dan backbone, koreksi Holm.
- Perbandingan ketahanan dibuat *matched* (backbone dan fold sama). Versi sementara sebelumnya membandingkan rata-rata yang tidak berpasangan (S3 belum punya MobileNetV2 pada fs4) dan bias.

### Perbaikan pengukuran
- **Latensi:** diganti dengan median `tf.function` (6,5–15 ms). Nilai tahap 1 (150–350 ms) berasal dari pemanggilan eager dan tidak valid; nilainya tetap ada di `table_main_group.csv` hanya untuk transparansi dan tidak dipakai di naskah.
- **Kecepatan S3:** bobot terbaik disimpan di RAM, cadangan ke Drive tiap 3 epoch (sebelumnya tiap epoch).

### Ketahanan Colab
- Supervisor latar belakang terlepas dari notebook, worker dengan daur ulang RAM, checkpoint atomik, PID lock, file stop.
- ZIP hasil dibuat dan diunduh otomatis di akhir setiap sesi (status `SEMENTARA`/`LENGKAP` dari `plan.json`).

### S0 yang tidak lengkap
- ResNet50 S0 fold 2 terhenti di epoch 21 ketika S0 dikeluarkan dari rencana; riwayatnya ada di `logs/history/` tetapi tidak ada baris hasilnya dan tidak dilaporkan. MobileNetV2 S0 tidak dijalankan.

## v0.2 — pengelompokan dari nama file (`filename-v2`)

- **Koreksi penting.** Versi v0.1 menebak potongan kain secara visual (kemiripan thumbnail antar-rotasi). Pemeriksaan terhadap nama file menunjukkan tebakan itu hanya benar untuk 19 dari 60 kelas, sehingga protokol "group" v0.1 sebenarnya masih bocor 14–49 %.
- Pengelompokan diganti dengan membaca nomor potongan dari nama file (`splits/groups.csv`).
- Seluruh hasil v0.1 diarsipkan dan **tidak dipakai**; semua job dijalankan ulang dengan fold yang benar.

## v0.1 — awal (tidak dipakai)

- Pipeline pertama dengan pengelompokan visual otomatis (`logs` lama: `group_detection.json`, tidak disertakan).
