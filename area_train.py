import glob
import os
import pickle
import random
import time
import math
import logging
import datetime
import tensorflow as tf

import numpy as np
import librosa
from tqdm import tqdm


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    #tf.random.set_seed(seed)

tf.enable_eager_execution()
#SEED = [111111,123456,0]
SEED=[999999,987654]
#SEED=[111111,123456,0,999999,987654]
attention_head = 4
attention_hidden = 32
area_height = 1
area_width = 1

import features
import model as MODEL

Epochs = 100
BATCH_SIZE = 32
learning_rate = 0.0001
T_stride = 2
T_overlop = T_stride / 2
overlapTime = {
    'neutral': 1,
    'happy': 1,
    'sad': 1,
    'angry': 1,
}
FEATURES_TO_USE = 'melspectrogram'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
featuresExist = True
impro_or_script = 'impro'
featuresFileName = 'features_{}_{}.pkl'.format(FEATURES_TO_USE, impro_or_script)
toSaveFeatures = True
WAV_PATH = "E:/Test/IEMOCAP/"
RATE = 16000

LABEL = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
}

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


def process_data(path, t=2, train_overlap=1, val_overlap=1.6, RATE=16000):
    path = path.rstrip('/')
    wav_files = glob.glob(path + '/*.wav')

    n = len(wav_files)
    train_files = []
    valid_files = []
    train_indices = list(np.random.choice(range(n), int(n * 0.8), replace=False))
    valid_indices = list(set(range(n)) - set(train_indices))
    for i in train_indices:
        train_files.append(wav_files[i])
    for i in valid_indices:
        valid_files.append(wav_files[i])

    print("constructing meta dictionary for {}...".format(path))
    train_X, train_y = train_data_process(train_files, LABEL_DICT1, RATE, t, train_overlap)

    val_dict = valid_data_process(valid_files, LABEL_DICT1, RATE, t, val_overlap)

    return train_X, train_y, val_dict


def valid_data_process(valid_files, LABEL_DICT1, RATE, t, val_overlap):
    val_dict = {}
    if (val_overlap >= t):
        val_overlap = t / 2
    for i, wav_file in enumerate(tqdm(valid_files)):
        label = str(os.path.basename(wav_file).split('-')[2])
        if (label not in LABEL_DICT1):
            continue
        if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            continue
        label = LABEL_DICT1[label]
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        index = 0
        if (t * RATE >= len(wav_data)):
            continue
        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(LABEL[label])
            index += int((t - val_overlap) * RATE)

        X1 = np.array(X1)
        val_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }
    return val_dict


def train_data_process(train_files, LABEL_DICT1, RATE, t, train_overlap):
    meta_dict = {}
    for i, wav_file in enumerate(tqdm(train_files)):
        label = str(os.path.basename(wav_file).split('-')[2])
        if (label not in LABEL_DICT1):
            continue
        if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            continue
        label = LABEL_DICT1[label]

        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        index = 0
        if (t * RATE >= len(wav_data)):
            continue

        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(LABEL[label])
            assert t - train_overlap > 0
            index += int((t - train_overlap) * RATE / overlapTime[label])

        X1 = np.array(X1)
        meta_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }
    print("building X, y...")
    train_X = []
    train_y = []
    for k in meta_dict:
        train_X.append(meta_dict[k]['X'])
        train_y += meta_dict[k]['y']
    train_X = np.row_stack(train_X)
    train_y = np.array(train_y)
    assert len(train_X) == len(train_y), "X length and y length must match! X shape: {}, y length: {}".format(
        train_X.shape, train_y.shape)
    return train_X, train_y


def train(SEED, area_width, area_height,AUG=True):
    setup_seed(SEED)
    MODEL_NAME = 'AUG_area{}x{}_seed{}'.format(area_width, area_height, SEED)
    data_dir = '/program/xumingke/IEMOCAP/'
    train_files = []
    train_files2 = []
    valid_files = []
    with open(data_dir + '/IEMOCAP_train_{}.csv'.format(SEED)) as f:
        fr = f.readlines()
        for line in fr:
            train_files.append(data_dir + '/' + line.split('\t')[2])
            if(AUG):
                for i in range(0):
                    train_files.append(data_dir + '/' + line.split('\t')[2]+'.'+str(i+1))
                for i in range(1):
                    train_files2.append(data_dir + '/' + line.split('\t')[2]+'.'+str(i+5))
    with open(data_dir + '/IEMOCAP_dev_{}.csv'.format(SEED)) as f:
        fr = f.readlines()
        for line in fr:
            train_files.append(data_dir + '/' + line.split('\t')[2])
            if(AUG):
                for i in range(0):
                    train_files.append(data_dir + '/' + line.split('\t')[2]+'.'+str(i+1))
                for i in range(1):
                    train_files2.append(data_dir + '/' + line.split('\t')[2]+'.'+str(i+5))            
    with open(data_dir + '/IEMOCAP_test_{}.csv'.format(SEED)) as f:
        fr = f.readlines()
        for line in fr:
            valid_files.append(data_dir + '/' + line.split('\t')[2])

    train_X, train_y = train_data_process(train_files, LABEL_DICT1, RATE, T_stride, T_overlop)
    train_X2,train_y2=train_data_process(train_files2, LABEL_DICT1, RATE, T_stride, T_overlop)
    train_y=tf.concat([train_y,train_y2],0)
    val_dict = valid_data_process(valid_files, LABEL_DICT1, RATE, T_stride, 1.6)
    feature_extractor = features.FeatureExtractor(rate=RATE)

    train_X_features = feature_extractor.get_features(FEATURES_TO_USE, train_X)
    train_X_features2 = feature_extractor.get_features(FEATURES_TO_USE, train_X2)

    valid_features_dict = {}
    for _, i in enumerate(val_dict):
        X1 = feature_extractor.get_features(FEATURES_TO_USE, val_dict[i]['X'])
        valid_features_dict[i] = {
            'X': X1,
            'y': val_dict[i]['y']
        }

    train_X_features = tf.expand_dims(train_X_features, -1)
    train_X_features2 = tf.expand_dims(train_X_features2, -1)
    train_X_features = tf.concat([train_X_features,train_X_features2],0)	
    train_X_features = tf.cast(train_X_features, tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_X_features, train_y)).shuffle(train_X_features.shape[0]).batch(BATCH_SIZE)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=1e-6)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    model = MODEL.AACNN(area_height, area_width)

    def train_step(images, labels):

        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    print('training...')
    logging.warning('training seed={}'.format(SEED))
    maxWA = 0
    maxUA = 0
    maxACC = 0
    for epoch in range(Epochs):
        # 在下一个epoch开始时，重置评估指标
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        # tq = tqdm(total=len(train_y))
        for step, (images, labels) in enumerate(train_ds):
            train_step(images, labels)

            # tq.update(BATCH_SIZE)
        # tq.close()
        template = 'Epoch {}, Loss: {}, Accuracy: {}\n'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              ))
        logging.warning(template.format(epoch + 1,
                                     train_loss.result(),
                                     train_accuracy.result() * 100,
                                     ))

        correct = 0
        label_correct = [0, 0, 0, 0]
        label_total = [0, 0, 0, 0]

        for _, i in enumerate(valid_features_dict):

            x, y = valid_features_dict[i]['X'], valid_features_dict[i]['y']
            x = tf.expand_dims(x, -1)
            x = tf.cast(x, tf.float32)
            y = np.array([y[0]])
            out = model(x)
            out = tf.reduce_mean(out, 0, keepdims=True)

            label_total[y[0]] += 1
            if (test_accuracy(y, out) > 0):
                correct += 1
                label_correct[y[0]] += 1
            test_accuracy.reset_states()

        label_acc = [label_correct[0] / label_total[0],
                     label_correct[1] / label_total[1],
                     label_correct[2] / label_total[2],
                     label_correct[3] / label_total[3]]
        UA = (label_acc[0] + label_acc[1] + label_acc[2] + label_acc[3]) / 4
        if (correct / len(valid_features_dict) > maxWA):
            maxWA = correct / len(valid_features_dict)
        if (UA > maxUA):
            maxUA = UA
        ACC = (correct / len(valid_features_dict)) + UA
        if (ACC > maxACC):
            print('saving model (WA:{},UA:{})\n'.format(correct / len(valid_features_dict), UA))
            logging.warning('saving model (WA:{},UA:{})\n'.format(correct / len(valid_features_dict), UA))
            model.save_weights('./models/{}'.format(MODEL_NAME))
            maxACC = ACC
        print('label_correct:{}\nUA:{}'.format(label_correct, label_acc))
        print('maxWA:{}\nmaxUA:{}'.format(maxWA, maxUA))
        logging.warning('label_correct:{}\nUA:{}'.format(label_correct, label_acc))
        logging.warning('maxWA:{}\nmaxUA:{}'.format(maxWA, maxUA))

    print('end training on seed:{}'.format(SEED))
    logging.warning('end training on seed:{}'.format(SEED))
    del model

    # model = MODEL.AACNN()
    #     # model.load_weights('./models/{}'.format(MODEL_NAME))
    #     #
    #     # result = []
    #     # correct = 0
    #     # for _, i in enumerate(valid_features_dict):
    #     #     x, y = valid_features_dict[i]['X'], valid_features_dict[i]['y']
    #     #     x = tf.expand_dims(x, -1)
    #     #     x = tf.cast(x, tf.float32)
    #     #     y = np.array([y[0]])
    #     #     out = model(x)
    #     #     out = tf.reduce_mean(out, 0, keepdims=True).numpy()
    #     #     if (np.argmax(out) == y):
    #     #         correct += 1
    #     #     result.append(out)
    #     # print(correct)
    #     # result = np.array(result)
    #     # np.save('./test/SERtest_{}.npy'.format(SEED), result)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    log_name = 'area3x3_AUG2.log2'.format(area_width, area_height)
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)

    for seed in SEED:
        train(seed, area_width, area_height,True)

