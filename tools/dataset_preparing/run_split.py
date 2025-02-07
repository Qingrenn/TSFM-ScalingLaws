import os
import  datasets
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
import logging

datasets.disable_caching()


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"run_{current_time}.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def reduce_size(dataset, remained_ratio, mode='sample'):
    assert mode in ['sample', 'time']
    if mode == 'sample':
        reduced_sample_num = int(np.ceil(len(dataset) * remained_ratio).item())
        remained_len = len(dataset[0]['target'])
        new_dataset = dataset.train_test_split(test_size=reduced_sample_num)["test"]
    else:
        remained_len = np.ceil(len(dataset[0]['target']) * remained_ratio)
        reduced_sample_num = len(dataset)
        def reduce_length_fn(example, remained_ratio):
            tgt = example["target"]
            example['target'] = tgt[-int(len(tgt) * remained_ratio):]
            return example
        
        new_dataset = dataset.map(
            reduce_length_fn,
            batched=False,
            fn_kwargs={"remained_ratio": remained_ratio}
        )
    
    return new_dataset

def train_test_split(dataset, test_ratio):
    
    def split_test_fn(example, test_len):
        tgt = example["target"]
        example['target'] = tgt[-test_len:]
        return example
    
    def split_train_fn(example, train_len):
        tgt = example["target"]
        example['target'] = tgt[:train_len]
        return example
    
    sample_len = len(dataset[0]['target'])
    if sample_len >= 192 * 2:
    
        test_len = max(int( sample_len * test_ratio ), 192)
        train_len = sample_len - test_len
        
        contetx_len = min(1000, sample_len - test_len)
        test_len = test_len + contetx_len
        
        train_dataset = dataset.map(split_train_fn, batched=False, fn_kwargs={"train_len": train_len})
        test_dataset = dataset.map(split_test_fn, batched=False, fn_kwargs={"test_len": test_len})
    
    elif len(dataset) > 1:
        test_sample_num = max(int(len(dataset) * test_ratio), 1)
        splitted_dataset = dataset.train_test_split(test_size=test_sample_num)
        train_dataset = splitted_dataset['train']
        test_dataset = splitted_dataset['test']

    else:
        raise ValueError('Dataset is too small to split')
    
    return train_dataset, test_dataset


def get_reduce_strategy(dataset, remained_ratio):
    '''
    ensure the reduced dataset have the minimum length of 192 * 2 and 1 sample
    '''
    min_time_len = 192 * 2
    min_sample_num = 1
    
    sum_tp = len(dataset[0]['target']) * len(dataset)
    if remained_ratio < (min_time_len / sum_tp):
        remained_ratio = max(remained_ratio, min_time_len / sum_tp)
        logger.info(f'dataset is too small, adjust remained_ratio to {remained_ratio}')
    
    sample_remained_ratio = max(remained_ratio, min_sample_num / len(dataset))
    if sample_remained_ratio * len(dataset) <= 1:
        time_remained_ratio = max(remained_ratio, min_time_len / len(dataset[0]['target']))
    else:
        time_remained_ratio = 1
    
    return {
        'sample': sample_remained_ratio,
        'time': time_remained_ratio
    }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', "--dataset", type=str)
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--remained_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.05)
    parser.add_argument('-o', '--output', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.dataset_dir is not None:
        for sub_dataset in os.listdir(args.dataset_dir):
            if os.path.exists(os.path.join(args.output, 'dataset_train', sub_dataset)):
                logger.info(f'{sub_dataset} already exists, skip')
                continue
                
            try:
                hf_dataset = datasets.load_from_disk(os.path.join(args.dataset_dir, sub_dataset))
                
                reduce_strategy = get_reduce_strategy(hf_dataset, args.remained_ratio)
                logger.info(f'{sub_dataset} reduce strategy: {reduce_strategy}')
                
                if reduce_strategy['sample'] < 1:
                    hf_dataset = reduce_size(hf_dataset, reduce_strategy['sample'], mode='sample')
                if reduce_strategy['time'] < 1:
                    hf_dataset = reduce_size(hf_dataset, reduce_strategy['time'], mode='time')
                
                train_dataset, test_dataset = train_test_split(hf_dataset, args.test_ratio)
                train_dataset.save_to_disk(f'{args.output}/dataset_train/{sub_dataset}')
                test_dataset.save_to_disk(f'{args.output}/dataset_test/{sub_dataset}')
            except Exception as e:
                logger.error(f'Error in processing {sub_dataset}: {e}')
                continue
            
    elif args.dataset is not None:
        hf_dataset = datasets.load_from_disk(args.dataset)
        reduce_strategy = get_reduce_strategy(hf_dataset, args.remained_ratio)
        logger.info(f'{args.dataset} reduce strategy: {reduce_strategy}')
        
        if reduce_strategy['sample'] < 1:
            hf_dataset = reduce_size(hf_dataset, reduce_strategy['sample'], mode='sample')
        if reduce_strategy['time'] < 1:
            hf_dataset = reduce_size(hf_dataset, reduce_strategy['time'], mode='time')
        
        dataset_name = args.dataset.split('/')[-1]
        train_dataset, test_dataset = train_test_split(hf_dataset, args.test_ratio)
        train_dataset.save_to_disk(f'{args.output}/dataset_train/{dataset_name}')
        test_dataset.save_to_disk(f'{args.output}/dataset_test/{dataset_name}')
    