# Pemetaan naskah → repositori

| Naskah | File di repositori |
|---|---|
| Gambar 1 (alur eksperimen) | `figures/fig1_workflow.png` (digambar penulis di draw.io) |
| Gambar 2 (akurasi bersih dan terkorupsi vs citra per kelas) | `figures/fig_fewshot_curve.png` ← `src/analysis.py` |
| Gambar 3 (akurasi per korupsi per backbone) | `figures/fig_robustness_by_backbone.png` ← `src/analysis.py` |
| Tabel 1 (strategi fine-tuning) | `docs/PROTOCOL.md` §3, `logs/config.json` |
| Tabel 2 (tahap 1, protokol group) | `results/table_main_group.csv`, `results/summary_all.csv` |
| Tabel 3 (protokol random vs group) | `results/table_leakage_inflation.csv`, kolom `leakage` di `results/results.csv` |
| Tabel 4 (akurasi bersih & rata-rata terkorupsi, 1/4/8 citra) | `results/table_fewshot.csv` |
| Tabel 5 (akurasi per korupsi) | `results/table_robustness_full.csv`, `results/table_robustness_by_strategy.csv` |
| Tabel 6 (perbandingan studi terdahulu) | literatur (tidak dari repositori ini) |
| Uji McNemar yang dikutip di teks | `results/stats_fewshot_mcnemar.csv`, `results/stats_pairwise.csv` |
| Waktu latih & latensi yang dikutip di teks | `results/table_fewshot.csv` |
| Angka yang dikutip di teks | `results/manuscript_numbers.json`, `docs/RESULTS.md` |

Gambar tambahan (tidak dipakai di naskah): `fig_accuracy_by_strategy.png`, `fig_robustness.png`, `fig_confusion_best.png`.

Naskah itu sendiri tidak disertakan sampai diterbitkan.
