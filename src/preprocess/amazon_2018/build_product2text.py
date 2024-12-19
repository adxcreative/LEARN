
import json
import os
import pickle

from tqdm import tqdm


def clean_txt(input_str):
    character_set_to_replace_str = {
        "&amp;": " and ",
        "&lt;": "",
        "&gt;": "",
        "&ndash": "",
        "&mdash;": "",
        "&ensp;": "",
        "&nbsp;": " ",
        "&shy;": "",
        "&copy;": "",
        "&trade;": "",
        "&reg;": "",
        "&quot;": ""
        }


    for character_set, replace_str in character_set_to_replace_str.items():
        input_str = input_str.replace(character_set, replace_str)
    return input_str
def get_text_description(meta_data):

    clean_meta_data = {}
    for key, value in tqdm(meta_data.items()):
        tmp = {}
        tmp['title'] = clean_txt(value['title'])
        tmp['brand'] = clean_txt(value['brand'])
        tmp['category'] = clean_txt(value['category'])
        clean_meta_data[key] = tmp
    return clean_meta_data


subdir = os.listdir('./amazon_2018/finetune_data/')

# ['Pet', 'Games', 'Instruments', 'Arts', 'Office', 'Scientific']
for i in subdir:
    meta = json.load(open(f'./amazon_2018/finetune_data/{i}/meta_data.json'))
    smap = json.load(open(f'./amazon_2018/finetune_data/{i}/smap.json'))

    clean_meta_data = get_text_description(meta)

    product2text = {}
    for key in smap.keys():
        if key in clean_meta_data:
            value = clean_meta_data[key]
            product2text[key] = f"Title: {value['title']}. Brand: {value['brand']}. Category: {value['category']}"
        else:
            print(f"missing {key} in {i}")

    print(f" total {len(product2text)} items in {i}")
    print(f" total {len(smap)} items needs in {i}")

    with open(f'./amazon_2018/finetune_data/{i}/product2text.dict', 'wb') as wf:
        pickle.dump(product2text, wf)
