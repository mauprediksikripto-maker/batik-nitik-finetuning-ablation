# @title Sel 2 - Rekap, uji statistik, grafik, ringkasan, lalu unduh ZIP hasil
# ---- Sel ini MANDIRI: aman dijalankan kapan saja, termasuk setelah runtime restart ----
import os, sys, json, glob, numpy as np, pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB and not os.path.isdir('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
PROJECT_DIR = globals().get('PROJECT_DIR') or ('/content/drive/MyDrive/BEEI_Batik' if IN_COLAB else os.environ.get('PROJECT_DIR', './BEEI_Batik'))
P = lambda *a: os.path.join(PROJECT_DIR, *a)
assert os.path.exists(P('logs', 'config.json')), 'Jalankan Sel 1 minimal sekali (setup & training) sebelum rekap.'
globals().update(json.load(open(P('logs', 'config.json'))))
RESULTS = P('results', 'results.csv')
try:
    import psutil as _ps
    _pidf = '/content/beei_supervisor.pid' if IN_COLAB else P('logs', 'beei_supervisor.pid')
    if os.path.exists(_pidf) and _ps.pid_exists(int(open(_pidf).read())) and \
            _ps.Process(int(open(_pidf).read())).status() != _ps.STATUS_ZOMBIE:
        print('Catatan: training masih berjalan di latar belakang -> rekap ini berisi hasil SEMENTARA.')
except Exception:
    pass
ENV = json.load(open(P('logs', 'environment.json')))
_cache = sorted(glob.glob(P('cache', 'batik_*.npz')))
_d = np.load(_cache[0], allow_pickle=True)
class_names = list(_d['class_names']); N_CLASSES = len(class_names); N_IMAGES = int(len(_d['y'])); del _d

try:
    # ===== 12. Rekap hasil (mean ± SD antar-fold) → results/summary_*.csv =====
    assert os.path.exists(RESULTS), 'Belum ada job selesai. Jalankan lagi (Run all) di sesi berikutnya.'
    res = pd.read_csv(RESULTS)
    agg = (res.groupby(['protocol', 'backbone', 'strategy'])
              .agg(acc_mean=('acc', 'mean'), acc_sd=('acc', 'std'), f1_mean=('f1_macro', 'mean'), f1_sd=('f1_macro', 'std'),
                   top5_mean=('top5', 'mean'), trainable_params=('trainable_params', 'first'), total_params=('total_params', 'first'),
                   train_time_s=('train_time_s', 'mean'), epochs=('epochs', 'mean'), latency_ms=('latency_ms', 'mean'),
                   leakage=('leakage', 'mean'), n_folds=('fold', 'nunique'))
              .reset_index())
    agg.to_csv(P('results', 'summary_all.csv'), index=False)
    fmt = lambda m, s: f'{100*m:.2f} ± {100*s:.2f}'
    tbl = agg[agg.protocol == 'group'].copy()
    tbl['Accuracy (%)'] = [fmt(m, s) for m, s in zip(tbl.acc_mean, tbl.acc_sd)]
    tbl['Macro-F1 (%)'] = [fmt(m, s) for m, s in zip(tbl.f1_mean, tbl.f1_sd)]
    tbl[['backbone', 'strategy', 'Accuracy (%)', 'Macro-F1 (%)', 'trainable_params', 'train_time_s', 'latency_ms']].to_csv(P('results', 'table_main_group.csv'), index=False)
    display(tbl[['backbone', 'strategy', 'Accuracy (%)', 'Macro-F1 (%)', 'trainable_params', 'train_time_s', 'latency_ms']])

    # Inflasi akibat kebocoran: random - group (strategi yang dijalankan di kedua protokol)
    piv = agg[agg.protocol.isin(['group', 'random'])].pivot_table(index=['backbone', 'strategy'], columns='protocol', values='acc_mean').dropna()
    if {'group', 'random'} <= set(piv.columns):
        piv['inflation_pp'] = 100 * (piv['random'] - piv['group'])
        piv.to_csv(P('results', 'table_leakage_inflation.csv'))
        display(piv)
except Exception as _e:
    print('[Bagian 12] gagal:', repr(_e))

try:
    # ===== 13. Uji statistik: McNemar (prediksi out-of-fold digabung) + paired t-test antar-fold, koreksi Holm =====
    from scipy.stats import ttest_rel, binomtest
    from itertools import combinations

    def oof(proto, bb, st, seed, key='y_pred'):
        fs = sorted(glob.glob(P('preds', f'{proto}_{bb}_f*_s{seed}_{st}.npz')))
        if len(fs) < N_FOLDS: return None
        d = [np.load(f) for f in fs]
        if any(key not in a.files for a in d): return None
        idx = np.concatenate([a['test_idx'] for a in d]); order = np.argsort(idx)
        return np.concatenate([a['y_true'] for a in d])[order], np.concatenate([a['y_pred' if key == 'y_pred' else key] for a in d])[order]

    def mcnemar_exact(yt, pa, pb):
        b = int(np.sum((pa == yt) & (pb != yt))); c = int(np.sum((pa != yt) & (pb == yt)))
        p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
        return b, c, p

    def holm(pvals):
        m = len(pvals); order = np.argsort(pvals); adj = np.empty(m); run = 0
        for rank, i in enumerate(order):
            run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
        return adj

    rows = []
    seed = SEEDS[0]
    for bb in BACKBONES:
        for a, b in combinations(STRATEGIES, 2):
            A, B = oof('group', bb, a, seed), oof('group', bb, b, seed)
            if A is None or B is None: continue
            nb, nc, p_mc = mcnemar_exact(A[0], A[1], B[1])
            ra = res[(res.protocol == 'group') & (res.backbone == bb) & (res.strategy == a)].sort_values('fold')['acc'].values
            rb = res[(res.protocol == 'group') & (res.backbone == bb) & (res.strategy == b)].sort_values('fold')['acc'].values
            p_t = ttest_rel(ra, rb).pvalue if len(ra) == len(rb) == N_FOLDS else np.nan
            rows.append(dict(backbone=bb, A=a, B=b, acc_A=np.mean(A[0] == A[1]), acc_B=np.mean(B[0] == B[1]),
                             A_only_correct=nb, B_only_correct=nc, p_mcnemar=p_mc, p_ttest=p_t))
    st = pd.DataFrame(rows)
    if len(st):
        st['p_mcnemar_holm'] = holm(st['p_mcnemar'].values)
        st['p_ttest_holm'] = holm(st['p_ttest'].fillna(1).values)
        st.to_csv(P('results', 'stats_pairwise.csv'), index=False)
        display(st.round(4))
except Exception as _e:
    print('[Bagian 13] gagal:', repr(_e))

try:
    # ===== 13b. Tahap 2: data sedikit (few-shot) & ketahanan terhadap korupsi citra =====
    import matplotlib.pyplot as plt
    fsr = res[res.protocol.astype(str).str.startswith('fs')].copy()
    fsa = pd.DataFrame(); rob = pd.DataFrame(); stfs = pd.DataFrame()
    if len(fsr):
        CORR = [f'acc_{c}' for c in CORRUPTIONS if f'acc_{c}' in fsr]
        fsa = (fsr.groupby(['shots', 'backbone', 'strategy'])
                  .agg(acc_mean=('acc', 'mean'), acc_sd=('acc', 'std'), f1_mean=('f1_macro', 'mean'), f1_sd=('f1_macro', 'std'),
                       corrupt_mean=('acc_corrupt_mean', 'mean'), corrupt_sd=('acc_corrupt_mean', 'std'),
                       train_time_s=('train_time_s', 'mean'), latency_ms=('latency_ms', 'mean'),
                       trainable_params=('trainable_params', 'first'), n_folds=('fold', 'nunique'))
                  .reset_index())
        fsa['robust_ratio'] = fsa.corrupt_mean / fsa.acc_mean
        fsa.to_csv(P('results', 'table_fewshot.csv'), index=False)
        print('Few-shot (mean antar-fold):'); display(fsa.round(4))
        # Perbandingan ADIL: hanya pasangan (backbone, fold) yang tersedia untuk SEMUA strategi pada jumlah data itu
        def matched(df):
            parts = []
            for k, d in df.groupby('shots'):
                keys = None
                for s_ in d.strategy.unique():
                    ks = set(map(tuple, d[d.strategy == s_][['backbone', 'fold']].values))
                    keys = ks if keys is None else keys & ks
                parts.append(d[[tuple(v) in keys for v in d[['backbone', 'fold']].values]])
            return pd.concat(parts) if parts else df
        fsm = matched(fsr)
        rob = fsm.groupby(['shots', 'strategy'])[CORR + ['acc', 'acc_corrupt_mean']].mean().reset_index()
        rob['n_backbone_fold'] = fsm.groupby(['shots', 'strategy']).size().values
        rob['backbones'] = fsm.groupby(['shots', 'strategy']).backbone.apply(lambda v: '+'.join(sorted(set(v)))).values
        rob.to_csv(P('results', 'table_robustness_by_strategy.csv'), index=False)
        fsr.groupby(['shots', 'backbone', 'strategy'])[CORR + ['acc', 'acc_corrupt_mean']].mean().reset_index() \
           .to_csv(P('results', 'table_robustness_full.csv'), index=False)
        print('Ketahanan per korupsi (rata-rata pasangan backbone-fold yang lengkap untuk semua strategi):'); display(rob.round(4))

        # McNemar eksak pada prediksi out-of-fold (960 citra), per kondisi; Holm per (shots, kondisi)
        rows = []
        for k in FS_SHOTS:
            for cond in ['clean'] + CORRUPTIONS:
                key = 'y_pred' if cond == 'clean' else f'pred_{cond}'
                for bb in BACKBONES:
                    for a, b in combinations(FS_STRATEGIES, 2):
                        A, B = oof(f'fs{k}', bb, a, SEEDS[0], key), oof(f'fs{k}', bb, b, SEEDS[0], key)
                        if A is None or B is None: continue
                        nb, nc, p_mc = mcnemar_exact(A[0], A[1], B[1])
                        rows.append(dict(shots=k, condition=cond, backbone=bb, A=a, B=b, acc_A=np.mean(A[0] == A[1]),
                                         acc_B=np.mean(B[0] == B[1]), A_only_correct=nb, B_only_correct=nc, p_mcnemar=p_mc))
        stfs = pd.DataFrame(rows)
        if len(stfs):
            stfs['p_holm'] = stfs.groupby(['shots', 'condition'])['p_mcnemar'].transform(lambda v: holm(v.values))
            stfs.to_csv(P('results', 'stats_fewshot_mcnemar.csv'), index=False)
            print('McNemar (kondisi bersih):'); display(stfs[stfs.condition == 'clean'].round(4))

        # Grafik 1: kurva akurasi vs jumlah citra latih per kelas
        fig, axes = plt.subplots(1, len(BACKBONES), figsize=(9, 3.4), dpi=300, sharey=True)
        for ax, bb in zip(np.atleast_1d(axes), BACKBONES):
            for s_ in FS_STRATEGIES:
                d = fsa[(fsa.backbone == bb) & (fsa.strategy == s_)].sort_values('shots')
                if not len(d): continue
                lab = s_.split('_')[0]
                l = ax.errorbar(d.shots, 100 * d.acc_mean, yerr=100 * d.acc_sd.fillna(0), marker='o', capsize=2, label=f'{lab} clean')
                ax.errorbar(d.shots, 100 * d.corrupt_mean, yerr=100 * d.corrupt_sd.fillna(0), marker='s', ls='--', capsize=2,
                            color=l[0].get_color(), alpha=.8, label=f'{lab} corrupted (mean of 7)')
            ax.set_title(bb, fontsize=9); ax.set_xticks(FS_SHOTS); ax.set_xlabel('Training images per class'); ax.grid(alpha=.3)
        np.atleast_1d(axes)[0].set_ylabel('Accuracy (%)')
        h_, l_ = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(h_, l_, loc='lower center', ncol=4, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(P('figures', 'fig_fewshot_curve.png')); plt.show()

        # Grafik 2: akurasi per jenis korupsi (rata-rata backbone) pada jumlah data terbanyak yang tersedia
        kmax = int(rob.shots.max()); d = rob[rob.shots == kmax].set_index('strategy')
        fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=300)
        conds = ['acc'] + CORR; xs = np.arange(len(conds)); w = 0.8 / max(1, len(d))
        for i, (s_, r_) in enumerate(d.iterrows()):
            ax.bar(xs + (i - (len(d) - 1) / 2) * w, [100 * r_[c] for c in conds], w, label=s_)
        ax.set_xticks(xs, ['clean'] + [c[4:] for c in CORR], fontsize=7); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 100)
        ax.set_title(f'{kmax} training image(s) per class (backbones: {d.backbones.iloc[0]})', fontsize=9)
        ax.legend(fontsize=7); ax.grid(axis='y', alpha=.3)
        fig.tight_layout(); fig.savefig(P('figures', 'fig_robustness.png')); plt.show()

        # Grafik 3: ketahanan per backbone (baris = jumlah data, kolom = backbone)
        rb = fsr.groupby(['shots', 'backbone', 'strategy'])[['acc'] + CORR].mean().reset_index()
        ks = sorted(rb.shots.unique())
        fig, axes = plt.subplots(len(ks), len(BACKBONES), figsize=(10, 2.7 * len(ks) + 0.4), dpi=300, sharey=True, squeeze=False)
        NICE = {'acc': 'Clean', 'acc_blur': 'Blur', 'acc_noise': 'Noise', 'acc_jpeg': 'JPEG', 'acc_dark': 'Dark',
                'acc_lowcontrast': 'Low contr.', 'acc_zoom': 'Zoom', 'acc_rot15': 'Rot. 15°'}
        for i, k in enumerate(ks):
            for j, bb in enumerate(BACKBONES):
                ax = axes[i][j]; d = rb[(rb.shots == k) & (rb.backbone == bb)].set_index('strategy')
                conds = ['acc'] + CORR; xs = np.arange(len(conds)); w = 0.8 / max(1, len(FS_STRATEGIES))
                for t, s_ in enumerate(FS_STRATEGIES):
                    if s_ in d.index:
                        ax.bar(xs + (t - (len(FS_STRATEGIES) - 1) / 2) * w, [100 * d.loc[s_, c] for c in conds], w,
                               label=s_.split('_')[0] + (' (frozen)' if s_.startswith('S1') else ' (full fine-tuning)'))
                ax.set_xticks(xs, [NICE.get(c, c) for c in conds], fontsize=7, rotation=45, ha='right')
                ax.tick_params(axis='y', labelsize=7)
                ax.set_ylim(0, 100); ax.grid(axis='y', alpha=.3)
                ax.set_title(f'{bb}, {int(k)} image{"s" if k > 1 else ""}/class', fontsize=8)
            axes[i][0].set_ylabel('Accuracy (%)', fontsize=8)
        h_, l_ = axes[0][0].get_legend_handles_labels()
        fig.legend(h_, l_, loc='lower center', ncol=2, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(P('figures', 'fig_robustness_by_backbone.png')); plt.show()
    else:
        print('Belum ada hasil tahap 2 (few-shot). Lanjutkan Sel 1.')
except Exception as _e:
    print('[Bagian 14] gagal:', repr(_e))

try:
    # ===== 14. Grafik (dari data hasil Anda) → figures/ =====
    import matplotlib.pyplot as plt
    g = agg[agg.protocol == 'group']
    assert len(g), 'Belum ada hasil protokol group. Lanjutkan training (Sel 1) dulu.'
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=300)
    w = 0.2; xs = np.arange(len(BACKBONES))
    for i, s in enumerate(STRATEGIES):
        sub = g[g.strategy == s].set_index('backbone').reindex(BACKBONES)
        ax.bar(xs + (i - 1.5) * w, 100 * sub.acc_mean, w, yerr=100 * sub.acc_sd, capsize=2, label=s.replace('_', ' '))
    ax.set_xticks(xs, BACKBONES); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 100); ax.legend(fontsize=7, ncol=4, loc='lower center')
    ax.grid(axis='y', alpha=.3); fig.tight_layout(); fig.savefig(P('figures', 'fig_accuracy_by_strategy.png')); plt.show()

    from sklearn.metrics import confusion_matrix
    best = g.sort_values('acc_mean', ascending=False).iloc[0]
    o = oof('group', best.backbone, best.strategy, SEEDS[0])
    if o is not None:
        cm = confusion_matrix(o[0], o[1], labels=np.arange(N_CLASSES))
        fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)
        im = ax.imshow(cm, cmap='Blues'); fig.colorbar(im, fraction=.046)
        ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title(f'{best.backbone} – {best.strategy}', fontsize=9)
        fig.tight_layout(); fig.savefig(P('figures', 'fig_confusion_best.png')); plt.show()
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(P('results', 'confusion_best.csv'))
    print('Konfigurasi terbaik (group):', best.backbone, best.strategy, f'{100*best.acc_mean:.2f}%')
except Exception as _e:
    print('[Bagian 15] gagal:', repr(_e))

try:
    # ===== 15. Ringkasan angka untuk naskah → results/manuscript_numbers.json =====
    summary = dict(environment=ENV, n_images=N_IMAGES, n_classes=int(N_CLASSES), folds=N_FOLDS, seeds=SEEDS,
                   best=dict(backbone=best.backbone, strategy=best.strategy, acc_mean=float(best.acc_mean), acc_sd=float(best.acc_sd),
                             f1_mean=float(best.f1_mean), f1_sd=float(best.f1_sd)),
                   main_table=g[['backbone', 'strategy', 'acc_mean', 'acc_sd', 'f1_mean', 'f1_sd', 'trainable_params', 'train_time_s', 'latency_ms']].to_dict('records'))
    if 'inflation_pp' in locals().get('piv', pd.DataFrame()).columns:
        summary['leakage_inflation_pp'] = piv.reset_index().to_dict('records')
    if len(fsa):
        summary['fewshot'] = fsa.to_dict('records')
        summary['robustness_by_strategy'] = rob.to_dict('records')
    if len(stfs):
        summary['fewshot_mcnemar_clean'] = stfs[stfs.condition == 'clean'].to_dict('records')
    json.dump(summary, open(P('results', 'manuscript_numbers.json'), 'w'), indent=2, default=float)
    print(json.dumps(summary['best'], indent=2))
    print('\nKirim file results/manuscript_numbers.json, results/*.csv, dan figures/*.png untuk mengisi naskah.')
except Exception as _e:
    print('[Bagian 16] gagal:', repr(_e))

# ===== 16. Kemas semua hasil -> ZIP di Drive + unduh otomatis =====
import zipfile, shutil, time as _t
_stamp = _t.strftime('%Y%m%d_%H%M')
_plan = set(json.load(open(P('logs', 'plan.json')))['jobs']) if os.path.exists(P('logs', 'plan.json')) else set()
_total = len(_plan)
_done = len(_plan & set(pd.read_csv(RESULTS)['job'])) if os.path.exists(RESULTS) else 0
_status = 'LENGKAP' if _done >= _total else f'SEMENTARA_{_done}dari{_total}'
ZIP_OUT = P(f'BEEI_Batik_hasil_{_status}_{_stamp}.zip')
_files = [f for sub in ['results', 'figures'] for f in glob.glob(P(sub, '*'))] + \
         glob.glob(P('logs', '*.json')) + glob.glob(P('logs', '*.csv')) + glob.glob(P('cache', 'groups.csv'))
with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in _files:
        z.write(f, os.path.relpath(f, PROJECT_DIR))
print(f'\nZIP hasil ({_status}): {ZIP_OUT}  ({len(_files)} file, {os.path.getsize(ZIP_OUT)/1e6:.2f} MB)')
if _done < _total:
    print(f'Job selesai {_done}/{_total}. Jalankan Sel 1 lagi di sesi berikutnya untuk melanjutkan.')
for f in sorted(_files):
    if not f.endswith('.history.csv'): print('  -', os.path.relpath(f, PROJECT_DIR))
if IN_COLAB:
    try:
        from google.colab import files as _colab_files
        _local_zip = os.path.join('/content', os.path.basename(ZIP_OUT))   # unduh dari disk lokal (lebih andal)
        shutil.copy(ZIP_OUT, _local_zip)
        _colab_files.download(_local_zip)
        print('Unduhan dimulai di browser. Bila tidak muncul, ambil ZIP di Google Drive: MyDrive/BEEI_Batik/')
    except Exception as _e:
        print('Unduh otomatis gagal (', repr(_e), '). Ambil ZIP di Google Drive: MyDrive/BEEI_Batik/')