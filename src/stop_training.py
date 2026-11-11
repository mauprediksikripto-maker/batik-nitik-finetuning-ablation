# @title Sel 3 - Hentikan training latar belakang (aman; checkpoint disimpan)
HENTIKAN = False   # ubah ke True lalu jalankan sel ini untuk menghentikan training
import os, sys, time
_run = '/content' if 'google.colab' in sys.modules else os.path.join(os.environ.get('PROJECT_DIR', './BEEI_Batik'), 'logs')
if HENTIKAN:
    open(os.path.join(_run, 'BEEI_STOP'), 'w').write(time.strftime('%H:%M:%S'))
    print('Sinyal berhenti dikirim. Training berhenti di akhir epoch berjalan (checkpoint disimpan).')
    print('Untuk melanjutkan nanti: jalankan Sel 1.')
else:
    print('Tidak melakukan apa-apa (HENTIKAN = False). Training tetap berjalan.')