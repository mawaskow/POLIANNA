'''
For use in subprocess to train the mhead models in loops without memory leakage between runs
'''
import os, json, sys, time, gc
import torch
from mhead_ner_ft import finetune_mhead_model

if __name__ == '__main__':
    model_name = sys.argv[1]
    r = int(sys.argv[2])
    model_save_addr = sys.argv[3]
    dsdct_dir = sys.argv[4]
    finetune_mhead_model(model_name, model_save_addr, dsdct_dir, r)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)