import glob
import os

import librosa
from tqdm import tqdm
import random
import numpy as np


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


LABEL_DICT1 = {
    '01': 'neutral',
    # '02': 'frustration',
    # '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    # '06': 'fearful',
    '07': 'happy',  # excitement->happy
    # '08': 'surprised'
}

impro_or_script='impro'
RATE=16000
T=2
def build_test_list(valid_files, LABEL_DICT1, RATE, t):
    testList=[]
    for i, wav_file in enumerate(tqdm(valid_files)):
        label = str(os.path.basename(wav_file).split('-')[2])
        if (label not in LABEL_DICT1):
            continue
        if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            continue
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        if (t * RATE >= len(wav_data)):
            continue
        testList.append(os.path.basename(wav_file))
    return testList

def process_IEMO():
    wavs = glob.glob('e:/test/iemocap/*.wav')
    transes = glob.glob('e:/test/iemo_t/trans/*.txt')
    write_list = []
    for wav in tqdm(wavs):
        wav_name = os.path.basename(wav)
        wav_name_split = wav_name.split('.')[0].split('-')
        if(wav_name_split[2] not in LABEL_DICT1):
            continue
        if ('script' in wav_name):
            txt_name = wav_name_split[0] + '_' + wav_name_split[1] + '_' + wav_name_split[-1].split('_')[0] + '.txt'
        else:
            txt_name = wav_name_split[0] + '_' + wav_name_split[1] + '.txt'
        trans_name = None
        for trans in transes:
            if (os.path.basename(trans) == txt_name):
                trans_name = trans
                break
        if (trans_name is not None):
            f_trans = open(trans_name)
            fr_trans = f_trans.readlines()
            FIND = False
            for l_trans in fr_trans:
                if (l_trans.split(' ')[0] == wav_name_split[0] + '_' + wav_name_split[1] + '_' + wav_name_split[-1]):
                    write_list.append((l_trans.split(' ')[0], l_trans.split(':')[-1].replace('\n',''), wav_name, wav_name_split[2]))
                    FIND = True
                    break
            if (FIND == False):
                print('Cannot find :' + wav_name)
            f_trans.close()
        else:
            print('Cannot find :' + txt_name)
    with open('IEMOCAP.csv', 'w') as f:
        for wl in write_list:
            for w in range(len(wl)):
                # f.write('\"' + wl[w] + '\"')
                f.write(wl[w])
                if (w < len(wl) - 1):
                    f.write('\t')
                else:
                    f.write('\n')


if __name__ == '__main__':
    SEED=[111111]
    # SEED=[123456,0,999999,987654]
    for seed in SEED:
        setup_seed(seed)
        process_IEMO()
        with open('IEMOCAP.csv', 'r') as f:
            fr = f.readlines()
        n = len(fr)
        trainAndDev_files = []
        train_files = []
        dev_files = []
        test_files = []
        trainAndDev_indices = list(np.random.choice(range(n), int(n * 0.8), replace=False))
        test_indices = list(set(range(n)) - set(trainAndDev_indices))
        for i in trainAndDev_indices:
            trainAndDev_files.append(fr[i])
        for i in test_indices:
            test_files.append(fr[i])
        n=len(trainAndDev_files)
        train_indices = list(np.random.choice(range(n), int(n * 0.875), replace=False))
        dev_indices = list(set(range(n)) - set(train_indices))
        for i in train_indices:
            train_files.append(trainAndDev_files[i])
        for i in dev_indices:
            dev_files.append(trainAndDev_files[i])

        data_dir = 'e:/test/iemocap'
        valid_files = []
        for line in test_files:
            valid_files.append(data_dir + '/' + line.split('\t')[2])
        test_wav=build_test_list(valid_files, LABEL_DICT1, RATE, T)
        bitest = []
        for line in test_files:
            if (line.split('\t')[2] in test_wav):
                bitest.append(line)
        with open('./IEMOCAP_bitest_{}.csv'.format(seed), 'w') as f:
            for line in bitest:
                f.write(line)

        with open('IEMOCAP_train_{}.csv'.format(seed), 'w') as f:
            for l in train_files:
                f.write(l)
        with open('IEMOCAP_dev_{}.csv'.format(seed), 'w') as f:
            for l in dev_files:
                f.write(l)
        with open('IEMOCAP_test_{}.csv'.format(seed), 'w') as f:
            for l in test_files:
                f.write(l)