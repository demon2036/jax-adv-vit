import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import webdataset as wds
import random
from glob import glob


def concat_dataset(dataset_path, data_type):
    images, labels, seq = None, None, None
    assert data_type == '50m'
    if data_type in ['1m', '10m']:
        pass
    elif data_type == '50m':

        print( )

        datas=glob(dataset_path)

        total_size = 50000000
        images = np.zeros(shape=(total_size, 32, 32, 3), dtype=np.uint8)
        labels = np.zeros(shape=(total_size,), dtype=np.uint64)

        # datas = ['50m_part1.npz', '50m_part2.npz', '50m_part3.npz', '50m_part4.npz']
        # datas = ['50m_part1.npz',]
        seq = np.arange(total_size)
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
        """

    return images, labels, seq


def test(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    images, labels, seq = concat_dataset(dataset_path=args.dataset_path, data_type=args.type)


    """
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
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",type=str,default='50m')
    parser.add_argument("--dataset-path")
    parser.add_argument("--shards-path")
    parser.add_argument("--seed",type=int,default=2036)
    # parser.add_argument("--hostname")
    # parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    test(args)
