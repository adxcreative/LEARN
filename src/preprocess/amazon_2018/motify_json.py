
import json
import os.path

data_list = ['Pet', 'Games', 'Instruments', 'Arts', 'Office', 'Scientific']



def replace_idx_to_product(data_dict, idx2product):

    for key, seq in data_dict.items():
        data_dict[key] = [idx2product[i] for i in seq]

    return data_dict


root_dir = './amazon_2018/finetune_data/'
for dataset in data_list:

    train_data = json.load(open(os.path.join(root_dir, f'{dataset}/train.json')))
    val_data = json.load(open(os.path.join(root_dir, f'{dataset}/val.json')))
    test_data = json.load(open(os.path.join(root_dir, f'{dataset}/test.json')))
    smap_data = json.load(open(os.path.join(root_dir, f'{dataset}/smap.json')))


    idx2product = dict([(val, key) for key, val in smap_data.items()])

    train_data_new = replace_idx_to_product(train_data, idx2product)
    val_data_new = replace_idx_to_product(val_data, idx2product)
    test_data_new = replace_idx_to_product(test_data, idx2product)

    with open(os.path.join(root_dir, f'{dataset}/train_product.json'), 'w') as f:
        json.dump(train_data_new, f)
    with open(os.path.join(root_dir, f'{dataset}/val_product.json'), 'w') as f:
        json.dump(val_data_new, f)
    with open(os.path.join(root_dir, f'{dataset}/test_product.json'), 'w') as f:
        json.dump(test_data_new, f)





