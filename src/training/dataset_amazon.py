from math import ceil
import os

import logging
import json
from dataclasses import dataclass
import random

import lmdb
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datetime import datetime


class AmazonDatasetPretrain(Dataset):
    def __init__(
            self,
            # split="train",
            train_stage=None,
            behavior_lmdb_path=None,
            item_emb_lmdb_path=None,
            max_one_year_length=256,
            max_one_month_length=20,
            sample_one_year_length=50,
            sample_one_month_length=4,
    ):
        super(AmazonDatasetPretrain, self).__init__()

        self.train_stage = train_stage

        self.item_emb_lmdb_path = item_emb_lmdb_path

        train_data = json.load(open(os.path.join(behavior_lmdb_path, 'train.json')))
        self.data = [seq for seq in train_data if len(seq) >= 5 and len(seq) <= 40]

        """
        训练数据百分位数统计
        10 5.0
        20 5.0
        30 6.0
        40 6.0
        50 7.0
        60 8.0
        70 9.0
        80 11.0
        90 15.0
        99 40
        100 3508.0
        """

        self.product2idx = json.load(open(os.path.join(behavior_lmdb_path, 'smap.json')))
        self.number_samples = len(self.data)

        self.item_env = lmdb.open(
            self.item_emb_lmdb_path,
            readonly=True,
            create=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.item_cursor = self.item_env.begin(buffers=True)
        logging.info("train LMDB file contains {} pairs.".format(self.number_samples))

        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples
        self.global_batch_size = 1  # will be modified to the exact global_batch_size after calling pad_dataset()

        self.max_one_year_length = max_one_year_length
        self.max_one_month_length = max_one_month_length
        self.sample_one_year_length = sample_one_year_length
        self.sample_one_month_length = sample_one_month_length

    def __del__(self):
        if hasattr(self, "behavior_env"):
            self.behavior_env.close()
        # if hasattr(self, "user_env"):
        #     self.user_env.close()
        if hasattr(self, "item_env"):
            self.item_env.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()

        index = index % self.number_samples
        pair = self.data[index]
        userid = index

        seq_len = len(pair)
        start = seq_len // 2
        target_idx = random.randint(start, seq_len - 1)
        history_info_list = pair[:target_idx]
        target_info_list = pair[target_idx:target_idx + 1]

        # sasrec loss
        history_item_emb_list = np.stack([self.itemid_to_emb(i) for i in history_info_list], axis=0)
        history_len = len(history_item_emb_list)

        history_item_emb_list = torch.tensor(history_item_emb_list, dtype=torch.float32)
        history_attention_mask = torch.ones((len(history_item_emb_list),), dtype=torch.long)

        if history_len < self.max_one_year_length:
            padding_size = self.max_one_year_length - history_len
            padding_embedding = torch.zeros((padding_size, 4096), dtype=torch.float32)
            padding_mask = torch.zeros((padding_size,), dtype=torch.long)

            # right padding
            history_item_emb_list = torch.cat([history_item_emb_list, padding_embedding], dim=0)
            history_attention_mask = torch.cat([history_attention_mask, padding_mask], dim=0)

        target_item_emb_list = torch.tensor(self.itemid_to_emb(target_info_list[0]), dtype=torch.float32)
        target_attention_mask = torch.ones((self.max_one_month_length,), dtype=torch.long)
        target_gt = int(self.product2idx[target_info_list[0]])

        return {
            "history_item_emb_list": history_item_emb_list,
            "history_attention_mask": history_attention_mask,

            "target_item_emb_list": target_item_emb_list,
            "target_attention_mask": target_attention_mask,
            "target_gt": target_gt,

        }

    def itemid_to_emb(self, itemid_string, is_tgt=False):
        # itemid_string, action_type@itemid@timestamp

        itemid = itemid_string
        item_info = pickle.loads(self.item_cursor.get("{}".format(itemid).encode("utf-8")).tobytes())
        return item_info

    def select_n_items_without_duplicates(self, item_id_list, n):
        """
        从 item_id_list 中选择 n 个 item, 首先选择不重复的 item，如果不够，再从重复的 item 中随机选择。
        Args:
            item_id_list:
            n:

        Returns:

        """

        # action_type, itemid, time_info = item_string.split("@")

        # item_id_list = [item_string.split("@")[1] for item_string in itemid_list]

        unique_items = list(set(item_id_list))  # 去除重复项并转换为列表
        sample_index_mask = torch.zeros((n,), dtype=torch.long)
        unique_selected_items = random.sample(unique_items, k=min(len(unique_items), n))  # 从不重复项中随机选择n个项
        # unique ratio of items

        # logging.info("dup ratio of items: {}".format(1 - len(unique_items) / len(item_id_list)))

        unique_selected_indices = []
        for item in unique_selected_items:
            indices = [i for i, x in enumerate(item_id_list) if x == item]  # 获取所有与选中项相等的索引
            index = random.choice(indices)  # 随机选择一个索引
            unique_selected_indices.append(index)

        sample_index_mask[:len(unique_selected_items)] = 1

        if len(unique_selected_items) < n:
            unique_selected_indices += random.choices(range(len(item_id_list)), k=n - len(unique_selected_items))

        assert len(unique_selected_indices) == n

        unique_selected_indices = torch.tensor(unique_selected_indices, dtype=torch.long)
        return unique_selected_indices, sample_index_mask

    def itemid_list_processor(self, item_info_list, max_length=100, sample_length=10, is_tgt=False):
        """

        Args:
            item_info_list: list of ['item_view@4454518686032@1658330878', 'item_buy@4665121810641@1658331214', 'live_item_click@4716873089483@1658332320'],

        Returns:

        """

        # import pdb; pdb.set_trace()
        """
        itemid_list

        """
        itemid_list = []
        # itemid 只用在计算对比loss中，因此只需要采样出的 item 的 itemid 而不需要所有的 item 的 itemid
        emb_tensor_list = []
        truncate_item_info_list = item_info_list[-max_length:]

        for i, itemid in enumerate(truncate_item_info_list):
            # torch.float64 -> torch.float32 due to the torch.zeros() dtype is float32
            emb_tensor_list.append(torch.tensor(self.itemid_to_emb(itemid)))
            itemid_list.append(int(self.product2idx[itemid]))

        len_item_list = len(emb_tensor_list)
        emb_tensor = torch.stack(emb_tensor_list, dim=0).to(torch.float32)
        attention_mask = torch.ones((len_item_list,), dtype=torch.long)

        # sample without replacement
        # sample_index = random.sample(range(max_length), k=sample_length)
        sample_index, sample_index_mask = self.select_n_items_without_duplicates(itemid_list, sample_length)
        itemid_tensor = torch.tensor([itemid_list[i] for i in sample_index], dtype=torch.long)

        if len_item_list < max_length:
            padding_size = max_length - len_item_list
            padding_embedding = torch.zeros((padding_size, 4096), dtype=torch.float32)
            padding_mask = torch.zeros((padding_size,), dtype=torch.long)

            # right padding
            emb_tensor = torch.cat([emb_tensor, padding_embedding], dim=0)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=0)

        return emb_tensor, attention_mask, sample_index, itemid_tensor, sample_index_mask


class AmazonDatasetFinetune(Dataset):
    def __init__(
            self,
            # split="train",
            train_stage=None,
            ft_data=None,
            behavior_lmdb_path=None,
            item_emb_lmdb_path=None,
            max_one_year_length=256,
            max_one_month_length=20,
            sample_one_year_length=50,
            sample_one_month_length=4,
    ):
        super(AmazonDatasetFinetune, self).__init__()

        self.train_stage = train_stage
        self.ft_data = ft_data

        self.item_emb_lmdb_path = item_emb_lmdb_path

        train_data = json.load(open(os.path.join(behavior_lmdb_path, 'train_product.json')))
        self.data = [(key, value) for key, value in train_data.items() if len(value) >= 5 and len(value) <= 40]
        self.product2idx = json.load(open(os.path.join(behavior_lmdb_path, 'smap.json')))

        with open(f'./amazon_2018/finetune_data/product2emb_{ft_data}.pkl', 'rb') as rf:
            self.product2emb = pickle.load(rf)

        self.idx2embedding = [torch.tensor(self.product2emb[productid], dtype=torch.float32) for productid, idx in
                              self.product2idx.items()]
        self.idx2embedding = torch.stack(self.idx2embedding, dim=0)

        self.number_samples = len(self.data)
        logging.info("train LMDB file contains {} pairs.".format(self.number_samples))

        self.dataset_len = self.number_samples
        self.global_batch_size = 1  # will be modified to the exact global_batch_size after calling pad_dataset()

        self.max_one_year_length = max_one_year_length
        self.max_one_month_length = max_one_month_length
        self.sample_one_year_length = sample_one_year_length
        self.sample_one_month_length = sample_one_month_length

    def __del__(self):
        if hasattr(self, "behavior_env"):
            self.behavior_env.close()
        if hasattr(self, "item_env"):
            self.item_env.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()

        index = index % self.number_samples
        userid, pair = self.data[index]

        seq_len = len(pair)
        start = 1
        target_idx = random.randint(start, seq_len - 1)
        history_info_list = pair[:target_idx]
        target_info_list = pair[target_idx:target_idx + 1]

        history_item_emb_list = np.stack([self.product2emb[i] for i in history_info_list], axis=0)
        history_len = len(history_item_emb_list)

        history_item_emb_list = torch.tensor(history_item_emb_list, dtype=torch.float32)
        history_attention_mask = torch.ones((len(history_item_emb_list),), dtype=torch.long)

        if history_len < self.max_one_year_length:
            padding_size = self.max_one_year_length - history_len
            padding_embedding = torch.zeros((padding_size, 4096), dtype=torch.float32)
            padding_mask = torch.zeros((padding_size,), dtype=torch.long)

            # right padding
            history_item_emb_list = torch.cat([history_item_emb_list, padding_embedding], dim=0)
            history_attention_mask = torch.cat([history_attention_mask, padding_mask], dim=0)

        # target_item_emb_list = self.idx2embedding
        target_attention_mask = torch.ones((len(self.idx2embedding), 1), dtype=torch.long)
        target_gt = int(self.product2idx[target_info_list[0]])

        return {
            "history_item_emb_list": history_item_emb_list,
            "history_attention_mask": history_attention_mask,
            "target_attention_mask": target_attention_mask,
            "target_gt": target_gt,
        }


class AmazonDatasetTestUser(Dataset):
    def __init__(
            self,
            behavior_lmdb_path,
            item_emb_lmdb_path,
            max_one_year_length=256,
    ):
        super(AmazonDatasetTestUser, self).__init__()

        self.behavior_lmdb_path = behavior_lmdb_path
        self.item_emb_lmdb_path = item_emb_lmdb_path
        self.max_one_year_length = max_one_year_length

        train_seq = json.load(open(os.path.join(self.behavior_lmdb_path, 'train_product.json')))
        val_seq = json.load(open(os.path.join(self.behavior_lmdb_path, 'val_product.json')))
        test_seq = json.load(open(os.path.join(self.behavior_lmdb_path, 'test_product.json')))

        smap = json.load(open(os.path.join(self.behavior_lmdb_path, 'smap.json')))
        self.product2idx = smap
        # self.idx2product = {v: k for k, v in smap.items()}

        data_seq = {}
        for idx, (userid, gt) in enumerate(test_seq.items()):
            data_seq[idx] = {'userid': userid,
                             'seq': train_seq[userid] + val_seq[userid],
                             'gt': gt}
        self.data_seq = data_seq
        self.dataset_len = len(self.data_seq)
        logging.info(f"test user dataset size: {self.dataset_len}")

        self.item_env = lmdb.open(
            self.item_emb_lmdb_path,
            readonly=True,
            create=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.item_cursor = self.item_env.begin(buffers=True)

    def __del__(self):
        if hasattr(self, "item_env"):
            self.item_env.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        pair = self.data_seq[index]
        userid = pair['userid']
        history_item_list = pair['seq']
        assert len(pair['gt']) == 1
        gt = int(self.product2idx[pair['gt'][0]])

        history_item_emb_list, history_attention_mask = self.itemid_list_processor(
            history_item_list, max_length=self.max_one_year_length
        )

        return {
            "userid": int(userid),
            "history_item_emb_list": history_item_emb_list,
            "history_attention_mask": history_attention_mask,
            "history_gt": gt,
        }

    def itemid_list_processor(self, itemid_list, max_length=100):

        history_emb_list = []
        history_item_list = itemid_list[-max_length:]
        len_item_list = len(history_item_list)
        history_attention_mask = torch.ones((len_item_list,), dtype=torch.long)

        for i, value in enumerate(history_item_list):
            # value = self.idx2product[value]  # 获取产品id
            history_emb_list.append(torch.tensor(self.itemid_to_emb(value), dtype=torch.float32))
        history_emb_tensor = torch.stack(history_emb_list, dim=0)

        if len_item_list < max_length:
            padding_size = max_length - len_item_list
            padding_embedding = torch.zeros((padding_size, 4096), dtype=torch.float32)
            padding_mask = torch.zeros((padding_size,), dtype=torch.long)
            # right padding
            history_emb_tensor = torch.cat([history_emb_tensor, padding_embedding], dim=0)
            history_attention_mask = torch.cat([history_attention_mask, padding_mask], dim=0)

        return history_emb_tensor, history_attention_mask

    def itemid_to_emb(self, itemid_string, is_tgt=False):
        # itemid_string, action_type@itemid@timestamp
        itemid = itemid_string
        item_info = pickle.loads(self.item_cursor.get("{}".format(itemid).encode("utf-8")).tobytes())
        return item_info


class AmazonDatasetTestItem(Dataset):
    def __init__(
            self,
            item_lmdb_path,
            item_emb_lmdb_path,
    ):
        super(AmazonDatasetTestItem, self).__init__()

        self.item_lmdb_path = item_lmdb_path
        self.item_emb_lmdb_path = item_emb_lmdb_path

        # open LMDB files
        smap = json.load(open(self.item_lmdb_path))
        self.product2idx = smap
        self.product_name = list(smap.keys())

        self.item_emb_env = lmdb.open(
            self.item_emb_lmdb_path,
            readonly=True,
            create=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.item_emb_cursor = self.item_emb_env.begin(buffers=True)

        self.dataset_len = len(self.product_name)
        logging.info(f"test item dataset size: {self.dataset_len}")

    def __del__(self):
        if hasattr(self, "item_env"):
            self.item_env.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        # index = index % self.number_samples

        itemid = index
        product_name = self.product_name[index]
        item_emb = pickle.loads(self.item_emb_cursor.get("{}".format(product_name).encode("utf-8")).tobytes())
        item_emb = torch.tensor(item_emb, dtype=torch.float32)

        return {
            "itemid": int(itemid),
            "item_emb": item_emb,
        }


def pad_dataset(dataset, global_batch_size):
    # edit dataset.__len__() of the dataset
    dataset.dataset_len = (
            ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    )
    dataset.global_batch_size = global_batch_size


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: Dataset
    epoch_id: int


def get_pretrain_dataset(
        args,
        max_one_year_length=256,
        max_one_month_length=20,
        sample_one_year_length=50,
        sample_one_month_length=4,
        epoch_id=0,
):
    db_path = args.train_data
    emb_path = args.emb_path

    dataset = AmazonDatasetPretrain(
        train_stage=args.train_stage,
        behavior_lmdb_path=db_path,
        item_emb_lmdb_path=emb_path,
        max_one_year_length=max_one_year_length,
        max_one_month_length=max_one_month_length,
        sample_one_year_length=sample_one_year_length,
        sample_one_month_length=sample_one_month_length,
    )

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    batch_size = args.batch_size
    global_batch_size = batch_size * torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    sampler.set_epoch(epoch_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers,
        sampler=sampler,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_finetune_dataset(
        args,
        max_one_year_length=256,
        max_one_month_length=20,
        sample_one_year_length=50,
        sample_one_month_length=4,
        epoch_id=0,
):
    db_path = args.train_data
    emb_path = args.emb_path

    dataset = AmazonDatasetFinetune(
        train_stage=args.train_stage,
        ft_data=args.ft_data,
        behavior_lmdb_path=db_path,
        item_emb_lmdb_path=emb_path,
        max_one_year_length=max_one_year_length,
        max_one_month_length=max_one_month_length,
        sample_one_year_length=sample_one_year_length,
        sample_one_month_length=sample_one_month_length,
    )

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    batch_size = args.valid_item_batch_size
    global_batch_size = batch_size * torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    sampler.set_epoch(epoch_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers,
        sampler=sampler,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_test_user_dataset(
        args,
        max_one_year_length=256,
        epoch_id=0,
):
    db_path = args.test_user_data
    emb_path = args.test_emb_lmdb

    dataset = AmazonDatasetTestUser(
        behavior_lmdb_path=db_path,
        item_emb_lmdb_path=emb_path,
        max_one_year_length=max_one_year_length,
    )

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    # batch_size = args.batch_size
    # global_batch_size = batch_size * torch.distributed.get_world_size()
    # pad_dataset(dataset, global_batch_size)

    # num_samples = dataset.dataset_len
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.valid_user_batch_size,
        pin_memory=False,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_test_item_dataset(
        args,
        epoch_id=0,
):
    db_path = args.test_item_data
    emb_path = args.test_emb_lmdb

    dataset = AmazonDatasetTestItem(
        item_lmdb_path=db_path,
        item_emb_lmdb_path=emb_path
    )

    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.valid_item_batch_size,
        pin_memory=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_amazon18_pretrain(
        args,
        epoch_id=0,
        max_one_year_length=256,
        max_one_month_length=20,
        sample_one_year_length=50,
        sample_one_month_length=4,
):
    data = {}

    data["train"] = get_pretrain_dataset(
        args,
        max_one_year_length=max_one_year_length,
        max_one_month_length=max_one_month_length,
        sample_one_year_length=sample_one_year_length,
        sample_one_month_length=sample_one_month_length,
        epoch_id=epoch_id,
    )

    data["test_user"] = get_test_user_dataset(
        args,
        max_one_year_length=max_one_year_length,
        epoch_id=epoch_id,
    )

    data["test_item"] = get_test_item_dataset(
        args,
        epoch_id=epoch_id,
    )

    return data


def get_amazon18_finetune(
        args,
        epoch_id=0,
        max_one_year_length=256,
        max_one_month_length=20,
        sample_one_year_length=50,
        sample_one_month_length=4,
):
    data = {}

    data["train"] = get_finetune_dataset(
        args,
        max_one_year_length=max_one_year_length,
        max_one_month_length=max_one_month_length,
        sample_one_year_length=sample_one_year_length,
        sample_one_month_length=sample_one_month_length,
        epoch_id=epoch_id,
    )

    data["test_user"] = get_test_user_dataset(
        args,
        max_one_year_length=max_one_year_length,
        epoch_id=epoch_id,
    )

    data["test_item"] = get_test_item_dataset(
        args,
        epoch_id=epoch_id,
    )

    return data

if __name__ == "__main__":
    pass
