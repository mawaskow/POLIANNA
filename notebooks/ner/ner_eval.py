'''
Evaluates the performance of the models.

Seqeval and Token scores
sghead and mhead
'''
import os
import evaluate
from datasets import DatasetDict
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import json

#############
# functions #
#############

############################## SGHEAD SEQEVAL ##############################

def sghead_getpreds(model_name, label_list, model_save_addr, dsdct_dir, r):
    device = torch.device("cuda")
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    model_tt = AutoModelForTokenClassification.from_pretrained(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}")
    model_tt = model_tt.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(list(dataset_dict['test']['text']), return_tensors="pt", padding=True, truncation=True)
    print(inputs)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model_tt(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    labels = list(dataset_dict['test']['ner_tags'])
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_predictions, true_labels

def seqeval_for_sghead(predictions, labels):
    seqeval = evaluate.load("seqeval")
    results_dict = {}
    results_dict["Overall"] = {"precision":[], "recall":[], "f1":[], "accuracy":[]}
    for ftr in ["Actor", "InstrumentType", "Objective", "Resource", "Time"]:
        results_dict[ftr] = {"precision":[], "recall":[], "f1":[], "number":[]}
    results = seqeval.compute(predictions=predictions, references=labels)
    for k in list(results):
        if k[:4]=="over":
            x, metric = k.split("_")
            results_dict['Overall'][metric].append(float(results[k]))
        else:
            for mtr in list(results[k]):
                results_dict[k][mtr].append(float(results[k][mtr]))
    return results_dict

def get_sghead_seqeval(model_name, label_list, model_save_addr, dsdct_dir, r, results_dir):
    predictions, labels = sghead_getpreds(model_name, label_list, model_save_addr, dsdct_dir, r)
    with open(f"{results_dir}/sghead_seqeval_{model_name.split('/')[-1]}_{r}_pandr.json", "w", encoding="utf-8") as f:
        json.dump({
            "pred": predictions,
            "real": labels
        }, f, indent=4)
    results_dict = seqeval_for_sghead(predictions, labels)
    with open(f"{results_dir}/sghead_seqeval_{model_name.split('/')[-1]}_{r}_results.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

############################## Post-processing of results ##############################

def visualize_run_results(mode, eval, model_name, r, results_dir, dsdct_dir):
    with open(f"{results_dir}/{mode}/{mode}_{eval}_{model_name.split('/')[-1]}_{r}_pandr.json", "r", encoding="utf-8") as f:
        pandr = json.load(f)
    dataset = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")['test']
    for idx, tok in enumerate(dataset[0]['tokens'][:]):
        print(f"{idx:03} | {tok:15} | "
            f"P:{pandr['pred'][0][idx]:2}  "
            f"R:{pandr['real'][0][idx]:2}  ")

########
# main #
########

def main():
    cwd = os.getcwd()
    results_dir = cwd+"/../results"
    sghead_dsdcts_dir = cwd+"/../inputs/sghead_dsdcts"
    sghead_models_dir = cwd+"/../models/sghead"
    with open(cwd+"/../inputs/sghead_ds/label_mapping.json", "r", encoding="utf-8") as f:
        label_list = json.load(f)
    #
    model_name = "microsoft/deberta-v3-base"
    r = 3
    #get_sghead_seqeval(model_name, label_list, sghead_models_dir, sghead_dsdcts_dir, r, results_dir+"/sghead")
    # postprocessing
    visualize_run_results("sghead", "seqeval", model_name, r, results_dir, sghead_dsdcts_dir)

if __name__=="__main__":
    main()