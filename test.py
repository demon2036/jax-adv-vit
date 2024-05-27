from pathlib import Path

import numpy as np
from tqdm import tqdm
import webdataset as wds
import random

images = np.zeros(shape=(50000000, 32, 32, 3), dtype=np.uint8)
labels = np.zeros(shape=(50000000,), dtype=np.uint64)

datas = ['50m_part1.npz', '50m_part2.npz', '50m_part3.npz', '50m_part4.npz']
# datas = ['50m_part1.npz',]

counter = 0
for data in datas:
    data_np = np.load(data)
    length = len(data_np['label'])
    images[counter:counter + length] = data_np['image']
    labels[counter:counter + length] = data_np['label']
    counter += length
    print(length)
    # print(len(images_np))

del data_np

seq = np.arange(50000000)
random.shuffle(seq)


shard_path = './cifar10-50m-wds'
shard_dir_path = Path(shard_path)
shard_dir_path.mkdir(exist_ok=True)
shard_filename = str(shard_dir_path / 'shards-%05d.tar')

shard_size = int(200 * 1000 ** 2)
with wds.ShardWriter(
        shard_filename,
        maxsize=shard_size,
) as sink, tqdm(seq) as pbar:
    for i, seq_number in enumerate(pbar):
        img = images[seq_number]
        label = labels[seq_number]

        sink.write({
            "__key__": str(i),
            "jpg.pyd": img,
            "cls": label,
            # "json": label,
        })
