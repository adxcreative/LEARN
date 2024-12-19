import os.path
import pickle

import lmdb
import numpy as np
from tqdm import tqdm

root_dir = './amazon_2018/item_embeddings'
file_list = [os.path.join(root_dir, i) for i in sorted(os.listdir(root_dir))]
print(f"processing file list {file_list}")

lmdb_dir = './amazon_2018/item_embeddings/item_baichuan_lmdb'
os.makedirs(lmdb_dir, exist_ok=True)
env_embedding = lmdb.open(lmdb_dir, map_size=1024 ** 4)
txn_embedding = env_embedding.begin(write=True)

write_idx = 0
for file in tqdm(file_list):
    for line in tqdm(open(file, 'r')):
        product_id, embedding_str = line.strip().split('\x01')

        embedding = [float(i) for i in embedding_str.split(',')]
        assert len(embedding) == 4096, f"embedding error, length is {len(embedding)}"
        embedding = np.array(embedding)

        txn_embedding.put(key="{}".format(product_id).encode('utf-8'), value=pickle.dumps(embedding))
        write_idx += 1
        if write_idx % 1000 == 0:
            txn_embedding.commit()
            txn_embedding = env_embedding.begin(write=True)

txn_embedding.put(key=b'num_samples', value="{}".format(write_idx).encode('utf-8'))
txn_embedding.commit()
env_embedding.close()

print(write_idx)