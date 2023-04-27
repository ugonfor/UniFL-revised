from . import BaseDataset, register_dataset

import pickle
import pandas as pd
import os
import torch
import numpy as np

import torch.utils.data

from typing import Dict, List, Any


from typing import List
def label2target(label: List):
    assert len(label) == 28
    levels = [1 for _ in range(22)]
    levels += [6,6,5,5,5,3]
    
    target = []
    for i in range(len(label)):
        if label[i] == -1:
            target += [0.5 for _ in range(levels[i])]
        else:
            if levels[i] == 1:
                target += [label[i]]
            else:
                tmp = [0 for _ in range(levels[i])]
                tmp[label[i]] = 1
                target += tmp
    assert len(target) == 52
    return target

# setting
class cfg:
    # cfg 바꿀 경우에는 model.py에 있는 config도 변경해주어야 함.
    dpe = False # use or not
    type_token = True # use or not
    pos_enc = True # use or not

    embed_dim = 128
    pred_dim = 128
    output_dim = 28

    dropout = 0.2
    n_layers = 4
    n_heads = 4
    max_word_len = 256
    max_seq_len = 256
    pred_pooling = "cls" # "cls" "mean" 중 선택

    dpe_index_size = 1 # 현재 구현 안한상태

    ratio = 5

    pred_target = 'labels'

    eventencoder = "transformer"

def padding_word(x, max_len):
    if len(x) < max_len: return x + [0] * (max_len - len(x))
    else: return x[:max_len]

def padding_seq(x, max_seq):
    if len(x) < max_seq: return x + [0 for _ in range(len(x[0]))] * (max_seq - len(x))
    else: return x[:max_seq]

@register_dataset("20233477_dataset")
class MyDataset20233477(BaseDataset):
    """
    TODO:
        create your own dataset here.
        Rename the class name and the file name with your student number
    
    Example:
    - 20218078_dataset.py
        @register_dataset("20218078_dataset")
        class MyDataset20218078(BaseDataset):
            (...)
    """

    def __init__(
        self,
        data_path: str, # data_path should be a path to the processed features
        # ...,
        **kwargs,
    ):
        super().__init__()

        data = ['mimiciii', 'mimiciv', 'eicu']

        self.base_path = [os.path.join(data_path, d) for d in data]
        self.pred_target = cfg.pred_target

        ## label data
        label = [ np.load(
            os.path.join(data_path, "label", self.pred_target + ".npy"),
            allow_pickle=True,
        ) for data_path in self.base_path ]
        
        self.num_datas = list(map(len, label)) # for dataset

        # concate all dataset
        label = np.concatenate(label)
        label = np.array( 
            list( map(lambda x: label2target(eval(x)), label.tolist()) )
        )
        self.label = torch.tensor(label, dtype=torch.long)

        print(f"[LOG] loaded {data} {self.num_datas} samples")
        
        ## for dataset
        self.data_dir = [ os.path.join(data_path, "npy") for data_path in self.base_path]
        

        ## other
        self.max_word_len = cfg.max_word_len
        self.max_seq_len = cfg.max_seq_len

        self.time_bucket_idcs = [
            idx for idx in range(4, 24)
        ]  # start range + bucket num + 1

        self.cls = 101

        
    def __getitem__(self, index):
        """
        Note:
            You must return a dictionary here or in collator so that the data loader iterator
            yields samples in the form of python dictionary. For the model inputs, the key should
            match with the argument of the model's forward() method.
            Example:
                class MyDataset(...):
                    ...
                    def __getitem__(self, index):
                        (...)
                        return {"data_key": data, "label": label}
                
                class MyModel(...):
                    ...
                    def forward(self, data_key, **kwargs):
                        (...)
                
        """
        data_idx = 0
        if self.num_datas[0] <= index:
            index -= self.num_datas[0]
            data_idx = 1
            if self.num_datas[1] <= index:
                index -= self.num_datas[1]
                data_idx = 2

        fname = f"{index}.npy"

        input_ids = np.load(
            os.path.join(self.data_dir[data_idx], "inputs_ids", fname), allow_pickle=True
        )
        type_ids = np.load(
            os.path.join(self.data_dir[data_idx], "type_ids", fname), allow_pickle=True
        )
        dpe_ids = np.load(
            os.path.join(self.data_dir[data_idx], "dpe_ids", fname), allow_pickle=True
        )
        label = self.label[index]

        out = {
            "input_ids": input_ids,
            "type_ids": type_ids,
            "dpe_ids": dpe_ids,
            "label": label
        }

        return out


    
    def __len__(self):
        return len(self.label)
    
    def collator(self, samples):
        """Merge a list of samples to form a mini-batch.
        
        Args:
            samples (List[dict]): samples to collate
        
        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        
        Note:
            You can use it to make your batch on your own such as outputting padding mask together.
            Otherwise, you don't need to implement this method.
        """
        #if 'dummy' in samples[0]: return 0
        #input_ids = np.array([s["input_ids"] for s in samples])
        #type_ids = np.array([s["type_ids"] for s in samples])
        #dpe_ids = np.array([s["dpe_ids"] for s in samples])

        output = dict()

        for key in ["input_ids", "type_ids", "dpe_ids"]:
            ids = np.array([s[key] for s in samples])

            # make organized data
            tmp = [
                list( 
                    map(len, input_id.tolist())
                ) for input_id in ids
            ]
            tmp = list(map( lambda x: max(x) if len(x) != 0 else 0, tmp))
            
            longest_word_len =  max(tmp)
            #print(longest_word_len)
            
            longest_seq_len = max(list(map(len, ids)))
            #print(longest_seq_len)
            
            word_pad_ids = [
                list(
                    map(lambda x: x + [0] * (longest_word_len - len(x)), input_id.tolist())
                ) for input_id in ids
            ]

            seq_pad_ids = list(
                map(lambda x: x + [[0 for _ in range(longest_word_len)] for __ in range(longest_seq_len - len(x))] , word_pad_ids)
            )

            ids = torch.tensor(np.array(seq_pad_ids))
            
            output[key] = ids
            
        if "label" in samples[0]:
            output["label"] = torch.stack([s["label"] for s in samples])
        return output
        