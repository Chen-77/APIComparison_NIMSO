"""
Load the entire dataset and format it and save it to disk
"""

import os
import sys
import json
from tqdm import tqdm
from pprint import pprint

from datasets import load_dataset


def process_data(data, mode: str):
    print(f"There are {len(data)} pieces of data in the {mode} set")

    data_dic_list = []
    for per_data in tqdm(data):
        data_dic_list.append({"text": per_data["text"], "label": per_data["label"]})

    # Writing to a json file
    dataset_folder = os.path.join(sys.path[0], 'dataset')
    os.makedirs(dataset_folder, exist_ok=True)
    fout_path = os.path.join(dataset_folder, mode + '.json')
    with open(fout_path, 'w', encoding='utf-8') as fout:
        json.dump(data_dic_list, fout, indent=4)


if __name__ == '__main__':
   
    dataset = load_dataset("tweet_eval", "emotion")
    print(dataset)  
    pprint(dataset['train'][0], width=1000)  
    print("=" * 100)

    train_data = dataset['train']
    process_data(train_data, 'train')

    validation_data = dataset['validation']
    process_data(validation_data, 'validation')

    test_data = dataset['test']
    process_data(test_data, 'test')



