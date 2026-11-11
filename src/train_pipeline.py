"""Pipeline training ablasi fine-tuning Batik Nitik (versi skrip dari Sel 1 notebook).

Jalankan: python src/train_pipeline.py   (lihat docs/REPRODUCIBILITY.md)
Variabel lingkungan: PROJECT_DIR, BEEI_SMOKE=1, PRETRAINED=imagenet|none, BEEI_BUDGET_MIN.
"""
# @title Sel 1 - Setup + training (aman diulang; otomatis melanjutkan job yang belum selesai)
# ===== 1. Konfigurasi =====
import os, time
WORKER = os.environ.get('BEEI_WORKER') == '1'          # True di dalam proses pekerja (training)
SESSION_START = float(os.environ.get('BEEI_SESSION_START', time.time()))
MAX_WORKER_RAM_GB = 6.0     # proses pekerja di-restart otomatis bila RAM-nya melewati batas ini
SESSION_BUDGET_MIN = float(os.environ.get('BEEI_BUDGET_MIN', 105))   # batas training per sesi (menit); sisa waktu untuk rekap. Sesi Colab Anda: maks 2 jam
SMOKE_TEST = os.environ.get('BEEI_SMOKE') == '1'         # True = uji cepat (sedikit epoch/job) untuk memastikan pipeline jalan
PROJECT_DIR = '/content/drive/MyDrive/BEEI_Batik'
IMG_SIZE = 224
BATCH = 32
N_FOLDS = 4
SEEDS = [42]                # tambah mis. [42, 123, 2024] bila waktu GPU cukup
BACKBONES = ['EfficientNetB0', 'ResNet50', 'MobileNetV2']
STRATEGIES = ['S0_scratch', 'S1_frozen', 'S2_partial', 'S3_full']
# Protokol random hanya untuk S1 & S3 (mengukur inflasi akibat kebocoran rotasi)
RANDOM_PROTOCOL_STRATEGIES = ['S1_frozen', 'S3_full']
PRETRAINED = os.environ.get('PRETRAINED', 'imagenet')  # bobot ImageNet
PARTIAL_FRACTION = 0.25     # porsi layer backbone teratas yang dibuka pada S2
EPOCHS = dict(head=60, finetune=15, scratch=30)   # head: pada fitur, sangat cepat
LR = dict(head=1e-3, partial=1e-4, full=3e-5, scratch=1e-3)
PATIENCE = dict(head=8, finetune=4, scratch=10)
BACKUP_EVERY_EPOCHS = 3     # frekuensi checkpoint anti-putus ke Drive (bobot terbaik disimpan di RAM)
# ---- Tahap 2: data sedikit (few-shot) + uji ketahanan ----
FS_SHOTS = [1, 4, 8]        # citra latih per kelas: 1 = 1 orientasi dari 1 potongan; 4 = 1 potongan penuh; 8 = 2 potongan (data penuh)
FS_STRATEGIES = ['S1_frozen', 'S3_full']   # S2 & sisa S0 dikeluarkan dari rencana (batas waktu GPU)
CORRUPTIONS = ['blur', 'noise', 'jpeg', 'dark', 'lowcontrast', 'zoom', 'rot15']
if SMOKE_TEST:
    EPOCHS = dict(head=2, finetune=2, scratch=2)
    PATIENCE = dict(head=1, finetune=1, scratch=1)
    BACKUP_EVERY_EPOCHS = 1

# ===== 2. Mount Drive & siapkan folder =====
import os, sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
elif os.environ.get('PROJECT_DIR'):
    PROJECT_DIR = os.environ['PROJECT_DIR']
elif not os.path.isdir('/content/drive/MyDrive'):
    PROJECT_DIR = './BEEI_Batik'
for sub in ['data', 'cache', 'results', 'preds', 'ckpt', 'backup', 'figures', 'logs']:
    os.makedirs(os.path.join(PROJECT_DIR, sub), exist_ok=True)
P = lambda *a: os.path.join(PROJECT_DIR, *a)
import json as _json
_json.dump(dict(BACKBONES=BACKBONES, STRATEGIES=STRATEGIES, RANDOM_PROTOCOL_STRATEGIES=RANDOM_PROTOCOL_STRATEGIES,
                N_FOLDS=N_FOLDS, SEEDS=SEEDS, IMG_SIZE=IMG_SIZE, FS_SHOTS=FS_SHOTS, FS_STRATEGIES=FS_STRATEGIES,
                CORRUPTIONS=CORRUPTIONS), open(P('logs', 'config.json'), 'w'), indent=1)
print('Project dir:', PROJECT_DIR)

# ===== 3. Import, GPU, mixed precision =====
import gc, glob, json, time, zipfile, shutil, platform, random
import numpy as np, pandas as pd
import tensorflow as tf, keras
from keras import layers
from PIL import Image
try:
    from IPython.display import display
except ImportError:
    display = print
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
if gpus:
    keras.mixed_precision.set_global_policy('mixed_float16')
ENV = dict(python=platform.python_version(), tensorflow=tf.__version__, keras=keras.__version__,
           gpu=(tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU') if gpus else 'CPU'))
json.dump(ENV, open(P('logs', 'environment.json'), 'w'), indent=2)
print(ENV)

# ===== 4. Siapkan dataset (cari di Drive -> unduh otomatis dari Mendeley Data bila tidak ada) =====
import subprocess, urllib.request
LOCAL_DATA = '/content/batik_nitik' if IN_COLAB else os.path.join(PROJECT_DIR, '_local_data')
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
ARC_EXT = ('.zip', '.rar', '.tar', '.gz', '.tgz', '.7z')
MENDELEY_ZIP = 'https://data.mendeley.com/public-api/zip/sgh484jxzy/download/3'

def find_images(root):
    return [f for f in glob.glob(os.path.join(root, '**', '*'), recursive=True) if f.lower().endswith(IMG_EXT)]

def extract(arc, dest):
    os.makedirs(dest, exist_ok=True); print('Ekstrak', arc)
    a = arc.lower()
    if a.endswith('.zip'):
        with zipfile.ZipFile(arc) as zf: zf.extractall(dest)
    elif a.endswith(('.tar', '.gz', '.tgz')):
        shutil.unpack_archive(arc, dest)
    else:                                              # .rar / .7z
        subprocess.run('apt-get -qq install -y unrar p7zip-full > /dev/null', shell=True)
        cmd = ['unrar', 'x', '-o+', arc, dest + '/'] if a.endswith('.rar') else ['7z', 'x', '-y', f'-o{dest}', arc]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    for inner in glob.glob(os.path.join(dest, '**', '*'), recursive=True):   # arsip bertingkat
        if inner.lower().endswith(ARC_EXT) and inner != arc:
            extract(inner, os.path.dirname(inner)); os.remove(inner)

def locate_dataset():
    if len(find_images(LOCAL_DATA)) >= 900:
        return LOCAL_DATA
    # 1) folder data/ di project: arsip atau citra langsung
    arcs = [f for f in glob.glob(P('data', '**', '*'), recursive=True) if f.lower().endswith(ARC_EXT)]
    if len(find_images(P('data'))) >= 900:
        return P('data')
    # 2) cari di seluruh MyDrive file/folder bernama *nitik* atau *batik*
    if not arcs and IN_COLAB:
        roots = glob.glob('/content/drive/MyDrive/**/*[Nn][Ii][Tt][Ii][Kk]*', recursive=True) + \
                glob.glob('/content/drive/MyDrive/**/*[Bb][Aa][Tt][Ii][Kk]*', recursive=True)
        for r in roots:
            if os.path.isdir(r) and len(find_images(r)) >= 900:
                print('Dataset ditemukan di', r); return r
        arcs = [r for r in roots if r.lower().endswith(ARC_EXT)]
    # 3) unduh otomatis dari Mendeley Data (Batik Nitik 960, CC BY 4.0)
    if not arcs:
        dst = P('data', 'batik_nitik_960.zip')
        print('Mengunduh dari Mendeley Data ...')
        try:
            req = urllib.request.Request(MENDELEY_ZIP, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=600) as r, open(dst + '.part', 'wb') as f:
                shutil.copyfileobj(r, f)
            os.replace(dst + '.part', dst)
            if not zipfile.is_zipfile(dst):
                os.remove(dst); raise RuntimeError('file unduhan bukan ZIP')
            arcs = [dst]
        except Exception as e:
            print('Unduh otomatis gagal:', e)
    for a in arcs:
        extract(a, LOCAL_DATA)
    return LOCAL_DATA

CACHE = P('cache', f'batik_{IMG_SIZE}.npz')
DATA_ROOT = locate_dataset() if not os.path.exists(CACHE) else None
files = sorted(find_images(DATA_ROOT)) if DATA_ROOT else ['(cache)']
if not files:
    print('Isi folder', P('data'), ':', os.listdir(P('data')))
    raise FileNotFoundError(
        'Dataset belum ada. Buka https://data.mendeley.com/datasets/sgh484jxzy/3 -> "Download All", '
        f'lalu unggah file ZIP-nya (tanpa diekstrak) ke {P("data")} dan jalankan ulang sel ini.')
if DATA_ROOT:
    LOCAL_DATA = DATA_ROOT
    print(len(files), 'citra ditemukan di', DATA_ROOT)
else:
    print('Memakai cache citra:', CACHE)

# ===== 5. Label kelas + cache citra 224x224 (sekali saja, disimpan di Drive) =====
CACHE = P('cache', f'batik_{IMG_SIZE}.npz')

def class_of(path):
    # Struktur umum: <root>/<nama_motif>/<file>.jpg. Jika semua file di satu folder, pakai prefiks nama file.
    parent = os.path.basename(os.path.dirname(path))
    return parent

if os.path.exists(CACHE):
    d = np.load(CACHE, allow_pickle=True)
    X, y, class_names, paths = d['X'], d['y'], list(d['class_names']), list(d['paths'])
else:
    labels = [class_of(f) for f in files]
    if len(set(labels)) < 10:   # fallback: nama file "<motif>_<n>.jpg" atau "<motif> (n).jpg"
        import re
        labels = [re.split(r'[_(\-]|\s\d', os.path.splitext(os.path.basename(f))[0])[0].strip() for f in files]
    class_names = sorted(set(labels))
    cidx = {c: i for i, c in enumerate(class_names)}
    X = np.zeros((len(files), IMG_SIZE, IMG_SIZE, 3), np.uint8)
    for i, f in enumerate(files):
        X[i] = np.asarray(Image.open(f).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    y = np.array([cidx[l] for l in labels], np.int32)
    paths = [os.path.relpath(f, LOCAL_DATA) for f in files]
    np.savez(CACHE, X=X, y=y, class_names=np.array(class_names), paths=np.array(paths))
N_CLASSES = len(class_names)
if N_CLASSES != 60: print('PERINGATAN: jumlah kelas', N_CLASSES, '!= 60. Periksa struktur folder dataset:', class_names[:5])
counts = np.bincount(y)
print(f'{len(y)} citra, {N_CLASSES} kelas, per kelas min={counts.min()} max={counts.max()}, RAM X={X.nbytes/1e6:.0f} MB')

# ===== 6. Kelompok "potongan kain sumber" (4 rotasi dari motif yang sama) - dibaca dari nama file =====
import re
# Tiap kelas = 4 potongan x 4 orientasi (0/90/180/270 derajat). Citra yang merupakan rotasi satu sama lain
# dikelompokkan otomatis (jarak minimum antar-rotasi pada thumbnail 32x32 grayscale).
GROUPS_FILE = P('cache', 'groups.csv')

def thumb(img):
    return np.asarray(Image.fromarray(img).convert('L').resize((32, 32), Image.BILINEAR), np.float32) / 255.0

def rot_dist(a, b):
    return min(float(np.mean((np.rot90(a, k) - b) ** 2)) for k in range(4))

def detect_groups(X, y, n_groups=4):
    groups = np.full(len(y), -1, np.int32); report = []
    T = np.stack([thumb(x) for x in X])
    for c in np.unique(y):
        idx = np.where(y == c)[0]; m = len(idx); size = m // n_groups
        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                D[i, j] = D[j, i] = rot_dist(T[idx[i]], T[idx[j]])
        lab = np.full(m, -1); g = 0
        for i in range(m):                             # greedy: citra + (size-1) tetangga rotasi terdekat
            if lab[i] >= 0: continue
            free = [j for j in np.argsort(D[i]) if lab[j] < 0 and j != i][:size - 1]
            lab[[i] + free] = min(g, n_groups - 1); g += 1
        groups[idx] = lab
        within = max(D[a, b] for a in range(m) for b in range(m) if a != b and lab[a] == lab[b])
        between = min(D[a, b] for a in range(m) for b in range(m) if lab[a] != lab[b])
        report.append(dict(cls=int(c), sizes=np.bincount(lab, minlength=n_groups).tolist(),
                           within_max=float(within), between_min=float(between), separable=bool(within < between)))
    return groups, report

GROUPING_VERSION = 'filename-v2'
FNAME_RE = re.compile(r'^\s*\d+\s+.+?\s(\d+)(?:_rotate_\d+)?\.[A-Za-z]+$')

def groups_from_filenames(paths, y):
    # Nama file Batik Nitik 960: "<no> <nama motif> <potongan 1-4>[_rotate_<90|180|270>].jpg"
    src = []
    for pth in paths:
        m = FNAME_RE.match(os.path.basename(pth))
        if not m: return None
        src.append(int(m.group(1)))
    src = np.array(src)
    groups = np.zeros(len(y), np.int32)
    for c in np.unique(y):                         # petakan nomor potongan -> 0..3 per kelas
        idx = np.where(y == c)[0]
        for gi, sv in enumerate(sorted(set(src[idx]))):
            groups[idx[src[idx] == sv]] = gi
    ok = all(np.bincount(groups[y == c], minlength=N_FOLDS).tolist() == [len(np.where(y == c)[0]) // N_FOLDS] * N_FOLDS
             for c in np.unique(y))
    return groups if ok else None

# Arsipkan hasil lama yang dibuat dengan pengelompokan versi lama (tidak valid)
_old = pd.read_csv(GROUPS_FILE) if os.path.exists(GROUPS_FILE) else None
if _old is not None and ('method' not in _old or _old['method'].iloc[0] != GROUPING_VERSION):
    arch = P(f'_arsip_tidak_valid_{time.strftime("%Y%m%d_%H%M")}')
    os.makedirs(arch, exist_ok=True)
    for sub in ['results', 'preds', 'ckpt', 'backup', 'figures']:
        if os.path.isdir(P(sub)) and os.listdir(P(sub)):
            shutil.move(P(sub), os.path.join(arch, sub)); os.makedirs(P(sub), exist_ok=True)
    for f in glob.glob(P('logs', '*.history.csv')): os.remove(f)
    shutil.move(GROUPS_FILE, os.path.join(arch, 'groups_lama.csv'))
    print('Hasil lama (pengelompokan versi lama) dipindah ke', arch, '- eksperimen diulang dengan pengelompokan dari nama file.')
    _old = None

if _old is not None:
    groups = _old['group'].values
else:
    groups = groups_from_filenames(paths, y)
    method = GROUPING_VERSION
    if groups is None:
        print('PERINGATAN: pola nama file tidak dikenali -> memakai deteksi rotasi otomatis (kurang andal).')
        groups, rep = detect_groups(X, y, n_groups=N_FOLDS); method = 'auto-rotation'
        json.dump(rep, open(P('logs', 'group_detection.json'), 'w'), indent=1)
    pd.DataFrame(dict(path=paths, label=y, group=groups, method=method)).to_csv(GROUPS_FILE, index=False)
    print('Pengelompokan potongan kain:', method)
print('Contoh kelompok kelas 0:', np.bincount(groups[y == 0]))
_rm = [re.search(r'_rotate_(\d+)', os.path.basename(p_)) for p_ in paths]
rot = np.array([int(m.group(1)) // 90 if m else 0 for m in _rm], np.int32)   # orientasi 0..3

# ===== 7. Pembagian fold (group = bebas bocor, random = acak per citra) =====
from sklearn.model_selection import StratifiedKFold

def make_folds(protocol, seed):
    # Return list of (train_idx, val_idx, test_idx) per fold.
    folds = []
    if protocol.startswith('fs'):                        # few-shot: subset data latih dari protokol group
        k_shot = int(protocol[2:])
        for tr, va, te in make_folds('group', seed):
            keep = []
            for c in np.unique(y[tr]):
                idx = tr[y[tr] == c]
                crops = sorted(set(groups[idx]))
                if k_shot <= 4:                          # 1 potongan kain, k orientasi pertama
                    sel = idx[groups[idx] == crops[0]]
                    sel = sel[np.argsort(rot[sel], kind='stable')][:k_shot]
                else:
                    sel = idx                            # 8 = kedua potongan (data penuh)
                keep.extend(sel.tolist())
            folds.append((np.array(sorted(keep)), va, te))
        return folds
    if protocol == 'group':
        for k in range(N_FOLDS):
            v = (k + 1) % N_FOLDS                       # kelompok validasi = kelompok berikutnya
            test = np.where(groups == k)[0]
            val = np.where(groups == v)[0]
            train = np.where((groups != k) & (groups != v))[0]
            folds.append((train, val, test))
    else:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        splits = [te for _, te in skf.split(np.zeros(len(y)), y)]
        for k in range(N_FOLDS):
            test = splits[k]; val = splits[(k + 1) % N_FOLDS]
            train = np.setdiff1d(np.arange(len(y)), np.concatenate([test, val]))
            folds.append((train, val, test))
    for tr, va, te in folds:                          # cek tidak ada irisan
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    return folds

def leakage_rate(train_idx, test_idx):
    # Proporsi citra uji yang punya 'saudara rotasi' di data latih.
    key = lambda i: (y[i], groups[i])
    tr = {key(i) for i in train_idx}
    return float(np.mean([key(i) in tr for i in test_idx]))

for proto in ['group', 'random'] + [f'fs{k}' for k in FS_SHOTS]:
    f = make_folds(proto, SEEDS[0])
    print(proto, [(len(a), len(b), len(c)) for a, b, c in f], 'leakage test->train:', [round(leakage_rate(a, c), 3) for a, b, c in f])

# ===== 8. Model, augmentasi, dan strategi fine-tuning =====
AUG = keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
], name='augment')

def make_ds(idx, training, seed):
    ds = tf.data.Dataset.from_tensor_slices((X[idx], y[idx]))
    if training:
        ds = ds.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH)
    if training:
        ds = ds.map(lambda a, b: (AUG(tf.cast(a, tf.float32), training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def preprocess_layer(name):
    if name == 'EfficientNetB0':
        return layers.Identity(name='prep')                      # EfficientNet Keras sudah menormalisasi 0-255 di dalam model
    if name == 'MobileNetV2':
        return layers.Rescaling(1 / 127.5, offset=-1.0, name='prep')
    if name == 'ResNet50':
        return layers.Lambda(keras.applications.resnet50.preprocess_input, name='prep')
    raise ValueError(name)

def build_model(name, pretrained):
    ctor = getattr(keras.applications, name)
    w = (None if PRETRAINED in ('', 'none', 'None') else PRETRAINED) if pretrained else None
    base = ctor(include_top=False, weights=w, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = preprocess_layer(name)(inp)
    x = base(x, training=False) if pretrained else base(x)   # pretrained: BN selalu mode inferensi
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(N_CLASSES, activation='softmax', dtype='float32')(x)
    return keras.Model(inp, out, name=name), base

def set_trainable(base, mode):
    # mode: frozen | partial | full. BatchNorm tetap beku agar statistik ImageNet tidak rusak oleh batch kecil.
    if mode == 'frozen':
        base.trainable = False; return
    base.trainable = True
    n = len(base.layers)
    cut = int(n * (1 - PARTIAL_FRACTION)) if mode == 'partial' else 0
    for i, l in enumerate(base.layers):
        l.trainable = (i >= cut) and not isinstance(l, layers.BatchNormalization)

def n_trainable(model):
    return int(sum(np.prod(v.shape) for v in model.trainable_weights))

# ===== 9. Training tahan putus (checkpoint atomik + resume ke Drive) =====
# Semua file ditulis ke file sementara lalu di-rename (atomik), sehingga putus koneksi
# saat menyimpan tidak pernah merusak checkpoint. Bobot terbaik & kondisi training (epoch,
# best val_loss, counter early stopping, learning rate) disimpan; Run all melanjutkan dari situ.
def _atomic_save_weights(model, path):
    tmp = path + '.tmp.weights.h5'
    model.save_weights(tmp); os.replace(tmp, path)

def _atomic_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f)
    os.replace(tmp, path)

def _try_load(model, path):
    try:
        model.load_weights(path); return True
    except Exception as e:
        print('   checkpoint rusak/tidak cocok, diabaikan:', os.path.basename(path), type(e).__name__)
        return False

class BudgetExceeded(Exception):
    pass

def minutes_left():
    _stop = os.environ.get('BEEI_STOP_FILE')
    if _stop and os.path.exists(_stop):
        return -1.0                                       # dihentikan pengguna (Sel 3)
    return SESSION_BUDGET_MIN - (time.time() - SESSION_START) / 60

class SafeCheckpoint(keras.callbacks.Callback):
    def __init__(self, best_path, latest_path, state_path, patience, every, state):
        super().__init__()
        self.best_path, self.latest_path, self.state_path = best_path, latest_path, state_path
        self.patience, self.every = patience, every
        self.best, self.wait = state.get('best', np.inf), state.get('wait', 0)
        self.prior, self.t0 = state.get('elapsed', 0.0), time.time()
        self.best_weights, self.best_dirty = None, False
    def on_epoch_end(self, epoch, logs=None):
        v = float(logs['val_loss'])
        if v < self.best:
            self.best, self.wait = v, 0
            self.best_weights, self.best_dirty = self.model.get_weights(), True   # di RAM (cepat)
        else:
            self.wait += 1
        stop = self.wait >= self.patience
        self.budget_hit = (not stop) and minutes_left() <= 0
        if stop or self.budget_hit or (epoch + 1) % self.every == 0:
            if self.best_dirty:                           # tulis bobot terbaik ke Drive hanya saat backup
                cur = self.model.get_weights(); self.model.set_weights(self.best_weights)
                _atomic_save_weights(self.model, self.best_path); self.model.set_weights(cur)
                self.best_dirty = False
            _atomic_save_weights(self.model, self.latest_path)
            lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
            _atomic_json(dict(epoch=epoch, best=self.best, wait=self.wait, lr=lr, stopped=stop,
                              elapsed=self.prior + time.time() - self.t0), self.state_path)
        if stop or self.budget_hit:
            self.model.stop_training = True

class EpochLogger(keras.callbacks.Callback):
    def __init__(self, path): super().__init__(); self.path = path
    def on_epoch_end(self, epoch, logs=None):
        row = dict(epoch=epoch, **{k: float(v) for k, v in (logs or {}).items()})
        pd.DataFrame([row]).to_csv(self.path, mode='a', header=not os.path.exists(self.path), index=False)

def fit_stage(model, tag, train_ds, val_ds, lr, epochs, patience, steps_per_epoch):
    best = P('ckpt', f'{tag}.best.weights.h5')
    bdir = P('backup', tag); os.makedirs(bdir, exist_ok=True)
    latest, state_f = os.path.join(bdir, 'latest.weights.h5'), os.path.join(bdir, 'state.json')
    hist = P('logs', f'{tag}.history.csv')
    state, initial_epoch = {}, 0
    if os.path.exists(state_f) and os.path.exists(latest):
        st_ = json.load(open(state_f))
        if _try_load(model, latest):
            state, initial_epoch, lr = st_, st_['epoch'] + 1, st_['lr']
            print(f'   melanjutkan {tag} dari epoch {initial_epoch} (best val_loss={st_["best"]:.4f})')
    elif os.path.exists(hist):
        os.remove(hist)                                   # mulai bersih bila tak ada state valid
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    t0, n_ep = time.time(), 0
    if not state.get('stopped') and initial_epoch < epochs:
        guard = SafeCheckpoint(best, latest, state_f, patience, BACKUP_EVERY_EPOCHS, state)
        cbs = [guard,
               keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=max(1, patience // 2), min_lr=1e-7),
               EpochLogger(hist)]
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, initial_epoch=initial_epoch, callbacks=cbs, verbose=2)
        n_ep = len(h.history.get('loss', []))
        if getattr(guard, 'budget_hit', False):
            raise BudgetExceeded(tag)                      # checkpoint aman di Drive; lanjut di sesi berikutnya
        if guard.best_weights is not None:
            model.set_weights(guard.best_weights)         # bobot terbaik dari RAM
        elif not (os.path.exists(best) and _try_load(model, best)):
            print('   peringatan: bobot terbaik tidak tersedia, memakai bobot terakhir')
    elif not (os.path.exists(best) and _try_load(model, best)):
        print('   peringatan: bobot terbaik tidak tersedia, memakai bobot terakhir')
    shutil.rmtree(bdir, ignore_errors=True)
    return state.get('elapsed', 0.0) + time.time() - t0, initial_epoch + n_ep

# ===== 10. Evaluasi =====
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, top_k_accuracy_score

from PIL import ImageFilter, ImageEnhance
def corrupt(imgs, name, seed=0):
    # Korupsi deterministik pada citra uji (uint8 HxWx3); satu tingkat keparahan per jenis.
    rng = np.random.default_rng(seed)
    out = np.empty_like(imgs)
    for i, a in enumerate(imgs):
        im = Image.fromarray(a)
        if name == 'blur':
            im = im.filter(ImageFilter.GaussianBlur(radius=2))
        elif name == 'noise':
            im = Image.fromarray(np.clip(a.astype(np.float32) + rng.normal(0, 25, a.shape), 0, 255).astype(np.uint8))
        elif name == 'jpeg':
            import io; buf = io.BytesIO(); im.save(buf, format='JPEG', quality=15); im = Image.open(buf).convert('RGB')
        elif name == 'dark':
            im = ImageEnhance.Brightness(im).enhance(0.5)
        elif name == 'lowcontrast':
            im = ImageEnhance.Contrast(im).enhance(0.5)
        elif name == 'zoom':
            w, h = im.size; c = int(w * 0.6); o = (w - c) // 2
            im = im.crop((o, o, o + c, o + c)).resize((w, h), Image.BILINEAR)
        elif name == 'rot15':
            im = im.rotate(15, resample=Image.BILINEAR)
        out[i] = np.asarray(im)
    return out

def evaluate_corruptions(model, test_idx):
    res, preds = {}, {}
    for name in CORRUPTIONS:
        Xc = corrupt(X[test_idx], name, seed=1)
        ds = tf.data.Dataset.from_tensor_slices(Xc).batch(BATCH).map(lambda a: tf.cast(a, tf.float32))
        pr = model.predict(ds, verbose=0).argmax(1)
        res[f'acc_{name}'] = float(np.mean(pr == y[test_idx])); preds[name] = pr.astype(np.int16)
    res['acc_corrupt_mean'] = float(np.mean([res[f'acc_{n}'] for n in CORRUPTIONS]))
    return res, preds

def evaluate(model, test_idx):
    probs = model.predict(make_ds(test_idx, False, 0), verbose=0)
    pred = probs.argmax(1); yt = y[test_idx]
    p, r, f, _ = precision_recall_fscore_support(yt, pred, average='macro', zero_division=0)
    # latensi inferensi per citra: graf terkompilasi, batch 1, median 50 citra setelah 5x warm-up
    infer = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    for _ in range(5):
        infer(tf.constant(X[test_idx[:1]], tf.float32)).numpy()
    ts = []
    for i in test_idx[:50]:
        t0 = time.perf_counter(); infer(tf.constant(X[i:i + 1], tf.float32)).numpy(); ts.append(time.perf_counter() - t0)
    lat = float(np.median(ts) * 1000)
    return dict(acc=accuracy_score(yt, pred), prec_macro=p, rec_macro=r, f1_macro=f,
                top5=top_k_accuracy_score(yt, probs, k=5, labels=np.arange(N_CLASSES)), latency_ms=lat), pred, probs

# ===== 11. Daftar job & eksekusi (bisa dihentikan/dilanjutkan kapan saja) =====
RESULTS = P('results', 'results.csv')

def append_result(row):
    # Kolom bisa bertambah (tahap 2 punya kolom korupsi): gabung lalu tulis ulang secara atomik.
    new = pd.DataFrame([row])
    if os.path.exists(RESULTS):
        new = pd.concat([pd.read_csv(RESULTS), new], ignore_index=True)
    tmp = RESULTS + '.tmp'; new.to_csv(tmp, index=False); os.replace(tmp, RESULTS)

def done_jobs():
    if not os.path.exists(RESULTS): return set()
    return set(pd.read_csv(RESULTS)['job'])

def job_id(proto, bb, st, fold, seed): return f'{proto}|{bb}|{st}|f{fold}|s{seed}'

# Urutan prioritas: hasil terpenting selesai lebih dulu bila sesi terputus
PRIORITY = [('group', 'S1_frozen'), ('group', 'S3_full'), ('group', 'S2_partial'),
            ('random', 'S1_frozen'), ('random', 'S3_full'),
            # tahap 2 (few-shot + ketahanan); urutan = prioritas bila sesi terputus
            ('fs1', 'S1_frozen'), ('fs1', 'S3_full'), ('fs4', 'S1_frozen'), ('fs4', 'S3_full'),
            ('fs8', 'S1_frozen'), ('fs8', 'S3_full')]
JOBS = []
for seed in SEEDS:
    for proto, st in PRIORITY:
        if proto == 'random' and st not in RANDOM_PROTOCOL_STRATEGIES: continue
        if proto.startswith('fs') and (int(proto[2:]) not in FS_SHOTS or st not in FS_STRATEGIES): continue
        if st not in STRATEGIES: continue
        for bb in BACKBONES:
            for fold in range(N_FOLDS):
                JOBS.append((proto, bb, st, fold, seed))
if SMOKE_TEST:
    JOBS = [j for j in JOBS if j[3] == 0 and j[1] == BACKBONES[0] and j[0] in ('group', 'fs1', 'fs8')]
if not WORKER:
    json.dump(dict(jobs=[job_id(*j) for j in JOBS]), open(P('logs', 'plan.json'), 'w'))
print(len(JOBS), 'job;', len(done_jobs()), 'sudah selesai')

def train_head_on_features(model, tr, va, head_w):
    # S1 (feature extraction): fitur GAP dihitung SEKALI dari backbone beku, lalu hanya head yang dilatih.
    # Jauh lebih cepat daripada melewatkan citra ke backbone setiap epoch.
    t0 = time.time()
    gap = next(l for l in model.layers if isinstance(l, layers.GlobalAveragePooling2D))
    fe = keras.Model(model.input, gap.output)
    feat = lambda idx: fe.predict(make_ds(idx, False, 0), verbose=0).astype(np.float32)
    ftr, fva = feat(tr), feat(va)
    inp = keras.Input((ftr.shape[1],)); h = layers.Dropout(0.3)(inp)
    out = layers.Dense(N_CLASSES, activation='softmax', dtype='float32')(h)
    head = keras.Model(inp, out)
    head.compile(optimizer=keras.optimizers.Adam(LR['head']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h_ = head.fit(ftr, y[tr], validation_data=(fva, y[va]), epochs=EPOCHS['head'], batch_size=BATCH, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=PATIENCE['head'], restore_best_weights=True)])
    model.layers[-1].set_weights(head.layers[-1].get_weights())
    model.save_weights(head_w + '.tmp.weights.h5'); os.replace(head_w + '.tmp.weights.h5', head_w)
    del fe, head
    return time.time() - t0, len(h_.history['loss'])

def run_job(proto, bb, st, fold, seed, wall_start=None):
    keras.utils.set_random_seed(seed + fold)
    tr, va, te = make_folds(proto, seed)[fold]
    steps = int(np.ceil(len(tr) / BATCH))
    train_ds, val_ds = make_ds(tr, True, seed + fold), make_ds(va, False, 0)
    stem = f'{proto}_{bb}_f{fold}_s{seed}'
    head_w = P('ckpt', f'{stem}_S1head.best.weights.h5')
    t_total, ep_total = 0.0, 0
    if st == 'S0_scratch':
        model, base = build_model(bb, pretrained=False)
        t, e = fit_stage(model, f'{stem}_S0', train_ds, val_ds, LR['scratch'], EPOCHS['scratch'], PATIENCE['scratch'], steps)
        t_total += t; ep_total += e
    else:
        model, base = build_model(bb, pretrained=True)
        set_trainable(base, 'frozen')
        if os.path.exists(head_w) and os.path.exists(head_w + '.json'):
            model.load_weights(head_w)                         # pakai ulang hasil S1 (head warm-up)
            meta = json.load(open(head_w + '.json'))
            t_total += meta['time']; ep_total += meta['epochs']
        else:
            t, e = train_head_on_features(model, tr, va, head_w)
            json.dump(dict(time=t, epochs=e), open(head_w + '.json', 'w'))
            t_total += t; ep_total += e
        if st in ('S2_partial', 'S3_full'):
            mode = 'partial' if st == 'S2_partial' else 'full'
            set_trainable(base, mode)
            t, e = fit_stage(model, f'{stem}_{st}', train_ds, val_ds, LR[mode], EPOCHS['finetune'], PATIENCE['finetune'], steps)
            t_total += t; ep_total += e
    metrics, pred, probs = evaluate(model, te)
    cpreds = {}
    if proto.startswith('fs'):
        cm_, cpreds = evaluate_corruptions(model, te); metrics.update(cm_)
    np.savez_compressed(P('preds', f'{stem}_{st}.npz'), test_idx=te, y_true=y[te], y_pred=pred, probs=probs.astype(np.float16),
                        **{f'pred_{k}': v for k, v in cpreds.items()})
    row = dict(job=job_id(proto, bb, st, fold, seed), protocol=proto, backbone=bb, strategy=st, fold=fold, seed=seed,
               shots=int(proto[2:]) if proto.startswith('fs') else (8 if proto == 'group' else np.nan),
               n_train=len(tr), n_val=len(va), n_test=len(te), leakage=leakage_rate(tr, te),
               trainable_params=n_trainable(model), total_params=int(model.count_params()),
               train_time_s=t_total, epochs=ep_total,
               wall_min=(time.time() - wall_start) / 60 if wall_start else np.nan, **metrics)
    append_result(row)
    # hemat ruang Drive: hapus checkpoint yang tidak dipakai lagi
    if st != 'S1_frozen':
        for f in glob.glob(P('ckpt', f'{stem}_{st[:2]}*.best.weights.h5')):
            os.remove(f)
    planned = {(j[0], j[2]) for j in JOBS}
    need = [s_ for s_ in ('S2_partial', 'S3_full') if (proto, s_) in planned]   # head S1 dipakai S2/S3
    fin = done_jobs()
    if all(job_id(proto, bb, s_, fold, seed) in fin for s_ in need):
        for f in glob.glob(head_w + '*'):
            os.remove(f)
    del model, base, train_ds, val_ds; keras.backend.clear_session(); gc.collect()
    return row

DEFAULT_MIN = {'S0_scratch': 5.0, 'S1_frozen': 0.6, 'S2_partial': 3.0, 'S3_full': 4.0}   # perkiraan awal (menit, T4)
def est_minutes(bb, st, proto='group'):
    m = DEFAULT_MIN[st] * (1.8 if bb == 'ResNet50' else 1.0)
    if os.path.exists(RESULTS):
        r = pd.read_csv(RESULTS); r = r[(r.backbone == bb) & (r.strategy == st) & (r.protocol == proto)]
        if len(r) and 'wall_min' in r: m = float(r.wall_min.median())
    if proto in ('fs1', 'fs4') and st != 'S1_frozen': m *= 0.5     # data latih kecil -> lebih cepat
    return m

def ram_status():
    import psutil
    return psutil.Process().memory_info().rss / 1e9, psutil.virtual_memory().available / 1e9

def worker_loop():
    # Exit code: 0 = semua selesai, 2 = anggaran waktu habis, 3 = minta restart (RAM)
    todo = [j for j in JOBS if job_id(*j) not in done_jobs()]
    for j in todo:
        if minutes_left() < 1:
            return 2
        print('\n==>', job_id(*j), time.strftime('%H:%M:%S'), f'| sisa anggaran {minutes_left():.0f} mnt', flush=True)
        t_job = time.time()
        try:
            r = run_job(*j, wall_start=t_job)
        except BudgetExceeded as e:
            print('   anggaran waktu habis saat', e, '- checkpoint tersimpan', flush=True)
            return 2
        rss, avail = ram_status()
        print(f"   acc={r['acc']:.4f} f1={r['f1_macro']:.4f} waktu={(time.time()-t_job)/60:.1f} mnt | "
              f"RAM pekerja {rss:.1f} GB, sisa RAM sistem {avail:.1f} GB", flush=True)
        if rss > MAX_WORKER_RAM_GB or avail < 2.5:
            print('   RAM tinggi -> proses pekerja di-restart untuk membebaskan memori', flush=True)
            return 3
    return 0

if WORKER:
    sys.exit(worker_loop())

# ---- Proses induk: training berjalan sebagai proses LATAR BELAKANG yang terlepas dari notebook ----
# Putus koneksi, tab ditutup, cell dihentikan, atau kernel restart TIDAK menghentikan training.
# Menjalankan Sel 1 lagi hanya MENYAMBUNG ke log (tidak membuat proses ganda).
import subprocess, base64, psutil
RUN_DIR = '/content' if IN_COLAB else P('logs')
PID_F, LOG_F = os.path.join(RUN_DIR, 'beei_supervisor.pid'), os.path.join(RUN_DIR, 'beei_train.log')
STOP_F = os.path.join(RUN_DIR, 'BEEI_STOP')
WORKER_FILE, SUP_FILE, REKAP_FILE = (os.path.join(RUN_DIR, f) for f in ('beei_worker.py', 'beei_supervisor.py', 'beei_rekap.py'))
NOISE = ('WARNING', 'I0000', 'E0000', 'W0000', 'oneDNN', 'cuda_', 'computation placer', 'absl', 'TF-TRT',
         'Unable to register', 'rebuild TensorFlow', 'cpu_feature_guard', 'UserWarning', 'self.data_adapter',
         'Contoh kelompok', 'leakage test->train', 'citra, ', 'Memakai cache', 'Pengelompokan', 'Project dir')

def supervisor_proc():
    try:
        pid = int(open(PID_F).read().strip())
        pr = psutil.Process(pid)
        if pr.status() != psutil.STATUS_ZOMBIE and any(SUP_FILE in c for c in pr.cmdline()):
            return pr
    except Exception:
        pass
    return None

todo = [j for j in JOBS if job_id(*j) not in done_jobs()]
sup = supervisor_proc()
if sup:
    print(f'Training SUDAH berjalan di latar belakang (PID {sup.pid}) -> menyambung ke log ...')
elif todo:
    print(f'Sisa {len(todo)} job, perkiraan {sum(est_minutes(j[1], j[2], j[0]) for j in todo):.0f} menit; '
          f'anggaran sesi {SESSION_BUDGET_MIN:.0f} menit')
    if os.path.exists(STOP_F): os.remove(STOP_F)
    _HERE = os.path.dirname(os.path.abspath(__file__))   # versi repo: salin modul saudara, bukan base64
    for _src, _dst in (('train_pipeline.py', WORKER_FILE), ('analysis.py', REKAP_FILE), ('supervisor.py', SUP_FILE)):
        shutil.copyfile(os.path.join(_HERE, _src), _dst)
    env = dict(os.environ, PROJECT_DIR=PROJECT_DIR, BEEI_SESSION_START=str(time.time()), PRETRAINED=PRETRAINED,
               TF_CPP_MIN_LOG_LEVEL='3', PYTHONUNBUFFERED='1', BEEI_STOP_FILE=STOP_F, BEEI_WORKER_FILE=WORKER_FILE,
               BEEI_REKAP_FILE=REKAP_FILE, BEEI_RESULTS=RESULTS, BEEI_TOTAL=str(len(JOBS)),
               BEEI_BUDGET_MIN=str(SESSION_BUDGET_MIN), MPLBACKEND='Agg')
    with open(LOG_F, 'a') as lf:
        lf.write(f'\n===== sesi baru {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n')
        _p = subprocess.Popen([sys.executable, '-u', SUP_FILE], env=env, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=RUN_DIR, start_new_session=True)
    open(PID_F, 'w').write(str(_p.pid))
    sup = supervisor_proc()
else:
    print('Semua job sudah selesai.')

def _show(chunk):
    for line in chunk.splitlines():
        if line.strip() and not any(n in line for n in NOISE):
            print(line, flush=True)

if sup or os.path.exists(LOG_F):
    pos = 0
    if os.path.exists(LOG_F):                           # tampilkan ~30 baris terakhir saat menyambung
        txt = open(LOG_F, errors='ignore').read()
        last = txt.rfind('===== sesi baru')
        tail = txt[last:] if last >= 0 else txt
        _show('\n'.join(tail.splitlines()[-30:])); pos = len(txt.encode())
    try:
        while True:
            alive = supervisor_proc() is not None
            with open(LOG_F, 'rb') as f:
                f.seek(pos); new = f.read(); pos += len(new)
            if new: _show(new.decode(errors='ignore'))
            if not alive:
                break
            time.sleep(3)
    except KeyboardInterrupt:
        print('\nTampilan log dihentikan. Training TETAP berjalan di latar belakang. '
              'Jalankan Sel 1 lagi untuk menyambung, atau Sel 3 untuk menghentikan training.')
        raise SystemExit
left = [j for j in JOBS if job_id(*j) not in done_jobs()]
print('\nSelesai:', len(JOBS) - len(left), '/', len(JOBS))
if left:
    print(f'Belum selesai {len(left)} job (~{sum(est_minutes(j[1], j[2], j[0]) for j in left):.0f} menit). '
          'Di sesi berikutnya jalankan Sel 1 lagi: otomatis melanjutkan.')