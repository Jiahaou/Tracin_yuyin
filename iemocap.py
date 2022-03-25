import glob
import os
import shutil
import numpy as np
LABEL_DICT1 = {  # 情绪标签文件
            '01': 'neutral',
            # '02': 'frustration',
            # '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            # '06': 'fearful',
            '07': 'happy',  # excitement->happy
            # '08': 'surprised'
        }
path="D:\XJH__tracin\IEMOCAP"
save_path="D:\XJH__tracin\Train"
wav_files = glob.glob("D:\XJH__tracin\IEMOCAP" + '/*.wav')
# for i,wav_file in enumerate(wav_files):
#     label = str(os.path.basename(wav_file).split('-')[2])
#     if label in LABEL_DICT1 and ("impro" in wav_file):
#         file=str(os.path.basename(wav_file).split('\\')[0])
#         dst = os.path.join(save_path, file)
#         shutil.move(wav_file, dst)
train_indices = list(np.random.choice(range(len(wav_files)), int(len(wav_files) * 0.8), replace=False))
for i,wav_file in enumerate(wav_files):
    if i in train_indices:
        file = str(os.path.basename(wav_file).split('\\')[0])
        dst = os.path.join(save_path, file)
        shutil.move(wav_file, dst)
