import glob
import itertools
import os
import random

import webdataset as wds
import numpy as np
from tqdm import tqdm
from pathlib import Path
import numpy as np

np.random.seed(1)


def make():
    shard_path = './shards_01'
    shard_dir_path = Path(shard_path)
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'shards-%05d.tar')

    size = 10000  #* 100
    maxcount = 100

    shard_size = int(50 * 1000 ** 2)
    with wds.ShardWriter(
            shard_filename, maxcount=maxcount,
            maxsize=shard_size,
    ) as sink, tqdm(range(size), total=size) as pbar:
        for i in pbar:
            sink.write({
                # "__key__": str(i),
                # 'x': str(i)
                "__key__": "%06d" % i,
                'x': "%06d" % i,
                # "json": label,
            })


if __name__ == '__main__':
    # make()
    ds = glob.glob('./shards_01/*')

    dataset = wds.WebDataset(ds,shardshuffle=True).mcached()

    """

    ops = [
        # itertools.cycle,
        wds.detshuffle(bufsize=10000,initial=10000),
        # wds.slice(jax.process_index(), None, jax.process_count()),
        # wds.split_by_worker,
        # # wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        # wds.detshuffle(),
        # wds.decode("pil", handler=wds.ignore_and_continue),
        # wds.to_tuple("jpg.pyd", "cls", handler=wds.ignore_and_continue),
        # partial(repeat_samples, repeats=3),
        # wds.map_tuple(train_transform, torch.tensor),
    ]

    # dataset = wds.WebDataset(urls=shard_path, handler=wds.ignore_and_continue)

    for op in ops:
        dataset = dataset.compose(op)
    """
    dataset = wds.DataPipeline(
        wds.SimpleShardList(ds,seed=1),
        # itertools.cycle,
        # wds.detshuffle(),

        wds.slice(0, None, 4),
        wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        # wds.split_by_worker,

    )

    """
    wds.tarfile_to_samples(handler=wds.ignore_and_continue),
    wds.detshuffle(),
    wds.decode("pil", handler=wds.ignore_and_continue),
    wds.to_tuple("jpg", "cls", handler=wds.ignore_and_continue),
    partial(repeat_samples, repeats=args.augment_repeats),
    wds.map_tuple(train_transform, torch.tensor),
    """


    # for data in dataset:
    #     print(data)
    #
    # print('\n' * 20)

    count = 0
    for data in dataset:
        print(data)

        count += 1
        if count == 100:
            break
    print('\n')
    count = 0
    for data in dataset:
        print(data)
        # break
        count += 1
        if count == 100:
            break
