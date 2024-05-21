import random

import webdataset as wds
import numpy as np
from tqdm import tqdm
from pathlib import Path
import numpy as np

part1 = np.load('1m.npz')

seq=np.arange(len(part1['label']))
random.shuffle(seq)




shard_path = './shards_01'
shard_dir_path = Path(shard_path)
shard_dir_path.mkdir(exist_ok=True)
shard_filename = str(shard_dir_path / 'shards-%05d.tar')

shard_size = int(50 * 1000 ** 2)
with wds.ShardWriter(
        shard_filename,
        maxsize=shard_size,
) as sink, tqdm(zip(part1['image'][seq], part1['label'][seq]),total=len(part1['label'])) as pbar:
    for i,(img,label) in enumerate(pbar):




        """
        sink.write({
            "__key__": str(i),
            "jpg.pyd": img,
            "cls": label,
            # "json": label,
        })
        """
