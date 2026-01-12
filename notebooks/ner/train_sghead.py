'''
For use in subprocess to train the sghead models in loops without memory leakage between runs
'''
import os, json, sys, time, gc
import torch
from sghead_ner_ft import finetune_sghead_model

params = {
            "num_epochs": 15,
            "lr": 3e-5,
            "weight_decay": 0.01,
            "batch_size":16,
            "num_warmup_steps":0,
            "patience": 3
        }

if __name__ == '__main__':
    model_name = sys.argv[1]
    r = int(sys.argv[2])
    model_save_addr = sys.argv[3]
    dsdct_dir = sys.argv[4]
    label_list = ['O', 'B-Actor', 'I-Actor', 'B-InstrumentType', 'I-InstrumentType', 'B-Objective', 'I-Objective', 'B-Resource', 'I-Resource', 'B-Time', 'I-Time']
    finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)