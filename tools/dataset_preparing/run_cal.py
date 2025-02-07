import pandas as pd
import numpy as np
import datasets
from argparse import ArgumentParser
from tqdm import tqdm
import os

parser = ArgumentParser()
parser.add_argument('-d', "--dataset_dir", type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
args = parser.parse_args()


table = pd.DataFrame(columns=['Dataset', 'Sample', 'Timepoint', 'Minlen', 'Maxlen', 'Meanlen'])
for i, sub_dataset in enumerate(os.listdir(args.dataset_dir)):
    try:
        print(f'Processing {sub_dataset}...')
        hf_dataset = datasets.load_from_disk(os.path.join(args.dataset_dir, sub_dataset))
        sample = len(hf_dataset)
        timepoint = 0
        minlen = 1e9
        maxlen = 0
        for entry in hf_dataset:
            tgt_len = len(entry['target'])
            timepoint += tgt_len
            minlen = min(minlen, tgt_len)
            maxlen = max(maxlen, tgt_len)
        
        table.loc[i] = [sub_dataset, sample, timepoint, minlen, maxlen, timepoint / sample]
    except Exception as e:
        print(f'Error in processing {sub_dataset}: {e}')
        continue

table.to_csv(args.output, index=False)



