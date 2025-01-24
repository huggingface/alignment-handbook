
import argparse
import datasets
from datasets import Dataset, DatasetDict, Features, load_dataset, load_from_disk
from glob import glob
import os
import math
import numpy as np
# logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="gather logp")
    
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="The local dir for input data.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="The local dir to save data.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=80,
        help="The number of processes to do data loading",
    )
    
    args = parser.parse_args()
    return args

def contain_none_logp(row):
        
    for k,v in row.items():
      if 'logp' in k and v is None:
        return False
    return True



if __name__ == "__main__":
    args = parse_args()
    print(f"Data processing args: {args}")
    model_name_list=['zephyr-7b-sft-full','Yi-6B-Chat','Meta-Llama-3-8B-Instruct']

    ds_list = [] 
    for m_idx,model_name in enumerate(model_name_list):
        ds_path = glob(os.path.join(args.data_root,model_name +'.json'))
        print(ds_path)
        ds = datasets.load_dataset('json',data_files=ds_path)['train']
        
        if m_idx==0:
          base_ds = ds.sort('id')
          base_ds = base_ds.rename_column("logp", f"{model_name}.logp")
          print(base_ds[0])
        else:
          tmp_ds = ds.sort('id')
          print(ds)
          base_ds = base_ds.add_column(name = f"{model_name}.logp",column=tmp_ds['logp'])

    print(base_ds)
    print(base_ds[0])

    filter_base_ds = base_ds.filter(contain_none_logp,num_proc=args.num_workers)
    
    print(filter_base_ds[0])
    print(f'{len(base_ds)-len(filter_base_ds)} data contain none logp....')
    

    filter_base_ds = filter_base_ds.map(lambda x: {'mean_log_llm_probs': np.mean([x[f'{m_name}.logp'] for m_name in model_name_list])}, num_proc=20)
    
    split_ds = filter_base_ds.train_test_split(test_size=0.005)
    print(ds)

    split_ds.save_to_disk(args.save_path)

