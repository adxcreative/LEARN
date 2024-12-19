
import argparse

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str
    )

    return parser.parse_args()


def get_item_embedding(single_itemid_batch, iteminfo_batch, tokenizer, model, device, wf):
    with torch.no_grad():
        with autocast(dtype=torch.float16):
            inputs = tokenizer(
                text=iteminfo_batch,
                #                 padding="max_length",
                truncation=True,
                max_length=500,
                return_tensors="pt",
            )
            # import pdb; pdb.set_trace()
            inputs = inputs.to(device)
            output = model(**inputs, output_hidden_states=True, return_dict=True)
            all_text_features = output['hidden_states'][-1]
            feature_dim = all_text_features.shape[2]
            # extract last non-padding tokens
            attention_masks = inputs['attention_mask']
            all_text_features = all_text_features * attention_masks.unsqueeze(dim=2)
            text_features = torch.sum(all_text_features, dim=1) / torch.sum(attention_masks, dim=1, keepdim=True)

            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            for itemid, text_feature in zip(single_itemid_batch, text_features):
                text_feature = text_feature.detach().cpu().numpy()
                text_feature_list = text_feature.tolist()
                text_feature_list = [str(value) for value in text_feature_list]
                text_feature_list_str = ','.join(text_feature_list)
                line = '\x01'.join([itemid, text_feature_list_str])
                wf.write(line + '\n')


if __name__ == '__main__':
    args = parse_args()

    with open(f'./amazon_2018/finetune_data/{args.dataset}/product2text.dict',
              'rb') as rf:
        product2text = pickle.load(rf)

    key_list = list(product2text.keys())
    print(f"total {len(key_list)} items")

    llm_checkpoint = 'baichuan-inc/Baichuan2-7B-Base'
    tokenizer = AutoTokenizer.from_pretrained(
        llm_checkpoint,
        cache_dir='./checkpoints/',
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        llm_checkpoint,
        cache_dir='./checkpoints/',
        trust_remote_code=True,
        device_map='auto',
    )

    batchsize = 1
    device = 'cuda:{}'.format(0)
    model = model.half()

    # import pdb; pdb.set_trace()
    filepath = f'./amazon_2018/item_embeddings/{args.dataset}.txt'
    wf = open(filepath, 'w')

    with torch.no_grad():
        single_itemid_batch = []
        single_itemidinfo_batch = []
        for itemid in tqdm(key_list):
            single_itemid_batch.append(itemid)
            single_itemidinfo_batch.append(product2text[itemid])
            if len(single_itemid_batch) < batchsize:
                continue
            # batch inference
            # import pdb; pdb.set_trace()
            # inputs = processor(text=single_itemidinfo_batch, padding=True, return_tensors="pt")
            # import pdb; pdb.set_trace()
            # '''
            get_item_embedding(single_itemid_batch, single_itemidinfo_batch, tokenizer, model, device, wf)
            single_itemid_batch = []
            single_itemidinfo_batch = []

    # deal with the last
    if len(single_itemidinfo_batch) > 0:
        get_item_embedding(single_itemid_batch, single_itemidinfo_batch, tokenizer, model, device, wf)

    wf.close()
