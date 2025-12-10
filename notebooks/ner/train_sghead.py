'''
For use in subprocess to train the models in loops without memory leakage between runs
'''
# oneft.py
import os, json, sys, time, gc
import torch
from sghead_ner_ft import finetune_sghead_model

def get_label_list():
    cwd = os.getcwd()
    with open(cwd+"/../inputs/sghead_ds/label_mapping.json", "r", encoding="utf-8") as f:
        label_list = json.load(f)
    return label_list

if __name__ == '__main__':
    model_name = sys.argv[1]
    r = int(sys.argv[2])
    model_save_addr = sys.argv[3]
    dsdct_dir = sys.argv[4]
    label_list = get_label_list()
    finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)