import os, sys, time, subprocess
W, RES, STOP = os.environ['BEEI_WORKER_FILE'], os.environ['BEEI_RESULTS'], os.environ['BEEI_STOP_FILE']
TOTAL, START, BUDGET = int(os.environ['BEEI_TOTAL']), float(os.environ['BEEI_SESSION_START']), float(os.environ['BEEI_BUDGET_MIN'])
def done():
    try:
        with open(RES) as f: return max(0, sum(1 for _ in f) - 1)
    except FileNotFoundError:
        return 0
print('[supervisor] mulai', time.strftime('%H:%M:%S'), f'| hasil tersimpan {done()} job', flush=True)
crashes = 0
while True:
    if os.path.exists(STOP): print('[supervisor] dihentikan pengguna (Sel 3)', flush=True); break
    if (time.time() - START) / 60 >= BUDGET - 1: print('[supervisor] anggaran waktu sesi habis', flush=True); break
    n0 = done()
    rc = subprocess.call([sys.executable, '-u', W], env=dict(os.environ, BEEI_WORKER='1'))
    if done() > n0: crashes = 0
    if rc in (0, 2): break
    if rc == 3: continue
    crashes += 1
    print(f'[supervisor] pekerja berhenti tak terduga (kode {rc}); restart ke-{crashes} dari checkpoint', flush=True)
    if crashes >= 3: print('[supervisor] gagal 3x tanpa kemajuan', flush=True); break
print('[supervisor] membuat rekap + ZIP di Drive ...', flush=True)
subprocess.call([sys.executable, '-u', os.environ['BEEI_REKAP_FILE']])
print('[supervisor] selesai', time.strftime('%H:%M:%S'), f'| hasil tersimpan {done()} job', flush=True)
