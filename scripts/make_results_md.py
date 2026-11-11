"""Bangun docs/RESULTS.md langsung dari CSV di results/ (tanpa salin angka manual).

Pemakaian:  python scripts/make_results_md.py            (dari root repositori)
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: pd.read_csv(os.path.join(ROOT, 'results', f))
pct = lambda x: f'{100 * x:.1f}'
CORR = ['blur', 'noise', 'jpeg', 'dark', 'lowcontrast', 'zoom', 'rot15']


def md_table(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    out += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(out)


def section_phase1():
    t = R('table_main_group.csv')
    t = t.assign(**{'Trainable params': t.trainable_params.map('{:,}'.format),
                    'Train time (s)': t.train_time_s.round(0).astype(int)})
    t = t[['backbone', 'strategy', 'Accuracy (%)', 'Macro-F1 (%)', 'Trainable params', 'Train time (s)']]
    res = R('results.csv')
    s0 = res[(res.protocol == 'group') & (res.backbone == 'ResNet50') & (res.strategy == 'S0_scratch')]
    note = ('ResNet50 S0 hanya 2 fold (per fold: ' + ' / '.join(pct(a) for a in s0.acc) + ' %). '
            if len(s0) else '')
    inf = R('table_leakage_inflation.csv')
    leak = res[res.protocol == 'random'].leakage
    return (f"## Tahap 1 — protokol group (bebas kebocoran), data penuh\n\n{md_table(t)}\n\n"
            f"{note}MobileNetV2 S0 tidak dijalankan (dikeluarkan dari rencana, batas waktu GPU).\n\n"
            f"**Inflasi kebocoran (random − group):** {inf.inflation_pp.min():.1f}–{inf.inflation_pp.max():.1f} pp "
            f"untuk semua backbone × S1/S3; kebocoran rotasi pada protokol random rata-rata "
            f"{pct(leak.mean())} % (rentang {pct(leak.min())}–{pct(leak.max())} %).\n\n"
            "> Catatan: kolom `latency_ms` di `table_main_group.csv` (tahap 1) diukur dengan pemanggilan eager "
            "dan **tidak valid**; latensi yang valid (median `tf.function`) ada di `table_fewshot.csv`.\n")


def section_fewshot():
    t = R('table_robustness_by_strategy.csv')
    rows = []
    for _, r in t.iterrows():
        rows.append([int(r.shots), r.strategy, pct(r.acc), pct(r.acc_corrupt_mean)] + [pct(r[f'acc_{c}']) for c in CORR])
    df = pd.DataFrame(rows, columns=['Citra/kelas', 'Strategi', 'Bersih (%)', 'Rata2 korupsi (%)'] + CORR)
    fs = R('table_fewshot.csv')
    piv = fs.pivot_table(index=['shots', 'backbone'], columns='strategy', values='corrupt_mean').reset_index()
    piv['S3 − S1 (pp)'] = (100 * (piv.S3_full - piv.S1_frozen)).round(1)
    piv = piv.rename(columns={'shots': 'Citra/kelas', 'backbone': 'Backbone'})
    piv['Citra/kelas'] = piv['Citra/kelas'].astype(int)
    piv['S1 korupsi (%)'] = piv.S1_frozen.map(pct)
    piv['S3 korupsi (%)'] = piv.S3_full.map(pct)
    piv = piv[['Citra/kelas', 'Backbone', 'S1 korupsi (%)', 'S3 korupsi (%)', 'S3 − S1 (pp)']]
    lat = fs.groupby('backbone').latency_ms.agg(['min', 'max']).round(1).reset_index()
    lat.columns = ['Backbone', 'Latensi min (ms)', 'Latensi maks (ms)']
    tt = fs.groupby('strategy').train_time_s.agg(['min', 'max']).round(0).astype(int).reset_index()
    tt.columns = ['Strategi', 'Waktu latih min (s)', 'Waktu latih maks (s)']
    return (f"## Tahap 2 — data sedikit + ketahanan korupsi (rata-rata 3 backbone × 4 fold)\n\n{md_table(df)}\n\n"
            f"### Selisih ketahanan per backbone (rata-rata 7 korupsi)\n\n{md_table(piv)}\n\n"
            f"### Biaya\n\n{md_table(tt)}\n\n{md_table(lat)}\n\n"
            "Latensi = median inferensi satu citra (`tf.function`, Tesla T4, mixed precision).\n")


def section_stats():
    m = R('stats_fewshot_mcnemar.csv')
    clean = m[m.condition == 'clean']
    cor = m[(m.condition != 'clean') & (m.p_holm < 0.05)]
    rows = []
    for s, g in cor.groupby('shots'):
        s1 = int((g.acc_A > g.acc_B).sum()); s3 = int((g.acc_B > g.acc_A).sum())
        rows.append([int(s), len(g), s1, s3])
    df = pd.DataFrame(rows, columns=['Citra/kelas', 'Uji signifikan (Holm p<0,05)', 'S1 menang', 'S3 menang'])
    return ("## Uji statistik (McNemar eksak, prediksi out-of-fold gabungan 960 citra, koreksi Holm)\n\n"
            f"Kondisi bersih: tidak ada perbedaan S1 vs S3 yang signifikan (p_holm minimum = {clean.p_holm.min():.3f}).\n\n"
            f"Kondisi terkorupsi:\n\n{md_table(df)}\n")


if __name__ == '__main__':
    txt = ('# Hasil eksperimen\n\nFile ini dibangkitkan otomatis oleh `scripts/make_results_md.py` dari `results/*.csv`. '
           'Jangan diedit manual.\n\n' + section_phase1() + '\n' + section_fewshot() + '\n' + section_stats())
    out = os.path.join(ROOT, 'docs', 'RESULTS.md')
    open(out, 'w', encoding='utf-8').write(txt)
    print(txt)
