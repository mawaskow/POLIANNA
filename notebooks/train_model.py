# oneft.py
import os, json, sys, time, gc
import torch
from datasets import DatasetDict
from finetuning import finetune_incmodel
from finetune import load_labelintdcts

int2label_dct, label2int_dct = load_labelintdcts()
hyper_dct = {"bn":{
                "epochs":5, 
                "lr":2E-5,
                "batch_size":16,
                "loss":None,
                "oversampling":None,
                "span_step":None,
                "int2label":int2label_dct["bn"],
                "label2int":label2int_dct["bn"]
            },
             "mc":{
                "epochs":15, 
                "lr":2E-5,
                "batch_size":16,
                "loss":None,
                "oversampling":None,
                "span_step":None,
                "int2label":int2label_dct["mc"],
                "label2int":label2int_dct["mc"]
            }
    }
loss_hyper = {"bn":{
            "epochs":5, 
            "lr":2E-5,
            "batch_size":16,
            "loss":[0.61, 2.7],
            "oversampling":None,
            "span_step":None,
            "int2label":int2label_dct["bn"],
            "label2int":label2int_dct["bn"]
        },
            "mc":{
            "epochs":15, 
            "lr":2E-5,
            "batch_size":16,
            "loss":[1.91, 0.55, 0.6, 5.48, 2.31, 0.72],
            "oversampling":None,
            "span_step":None,
            "int2label":int2label_dct["mc"],
            "label2int":label2int_dct["mc"]
        }
    }
os_hyper = {"bn":{
            "epochs":5, 
            "lr":2E-5,
            "batch_size":16,
            "loss":None,
            "oversampling":"auto",
            "span_step":None,
            "int2label":int2label_dct["bn"],
            "label2int":label2int_dct["bn"]
        },
            "mc":{
            "epochs":15, 
            "lr":2E-5,
            "batch_size":16,
            "loss":None,
            "oversampling":"auto",
            "span_step":None,
            "int2label":int2label_dct["mc"],
            "label2int":label2int_dct["mc"]
        }
    }

if __name__ == '__main__':
    r = int(sys.argv[1])
    mode = sys.argv[2]
    letter = sys.argv[3]
    input_dir = sys.argv[4]
    output_dir = sys.argv[5]
    model_name = sys.argv[6]
    comd = sys.argv[7]
    i = 0
    while not torch.cuda.is_available():
        time.sleep(3)
        i+=1
        if i>5:
            sys.exit()
    ds = DatasetDict.load_from_disk(f"{input_dir}/ds_{r}_{mode}")
    if comd == "loss":
        hyper = loss_hyper[mode]
    elif comd == "os":
        hyper = os_hyper[mode]
    else:
        hyper = hyper_dct[mode]
    hyper["r"]=r
    metrics = finetune_incmodel(
        ds,
        int2label_dct[mode],
        label2int_dct[mode],
        mode,
        model_name=model_name,
        dev='cuda',
        output_dir=output_dir,
        hyperparams=hyper
    )
    with open(os.path.join(output_dir, f"run_details_{r}.json"), "w") as f:
        json.dump({mode: hyper}, f, indent=4)

    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)