import nlpaug.augmenter.audio as naa
import librosa
import glob
from tqdm import tqdm
import os
wlist=glob.glob(r'../IEMOCAP/*.wav')
targetDir='../IEMOCAP/'
aug = naa.VtlpAug(16000, zone=(0.0, 1.0), coverage=1, duration=None, fhi=4800, factor=(0.8, 1.2))
for w in tqdm(wlist):
    for i in range(7):
        wav,_=librosa.load(w,16000)
        wavAug=aug.augment(wav)
        wavName=os.path.basename(w)
        librosa.output.write_wav(targetDir+wavName+'.'+str(i+1),wavAug,16000)
