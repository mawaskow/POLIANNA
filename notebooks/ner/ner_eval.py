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
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from collections import Counter
from sghead_ner_ft import SgheadDataset, sghead_collate, sghead_evaluate_model
from mhead_ner_ft import MheadDataset, MheadTokenClassifier, mhead_collate, mhead_evaluate_model

#############
# functions #
#############

def extract_results(raw_res_dct, mode="sghead", eval_type="seqeval"):
    results_dict = {}
    results_dict["Overall"] = {}
    for ftr in ["Actor", "InstrumentType", "Objective", "Resource", "Time"]:
        results_dict[ftr] = {}
    if eval_type == "seqeval":
        if mode =="sghead":
            for k in list(raw_res_dct):
                if k[:4]=="over":
                    x, metric = k.split("_")
                    results_dict['Overall'][metric]=float(raw_res_dct[k])
                else:
                    for mtr in list(raw_res_dct[k]):
                        results_dict[k][mtr]=float(raw_res_dct[k][mtr])
        elif mode == "mhead":
            for k in list(raw_res_dct):
                if k[:4]=="over":
                    pass
                else:
                    for mtr in list(raw_res_dct[k]):
                        results_dict[k][mtr]=float(raw_res_dct[k][mtr])
    return results_dict

def evaluate_model(mode, model_name, model_save_addr, dsdct_dir, r, batch_size=16, dropout=0.1):
    if mode == "sghead":
        label_list = ['O', 'B-Actor', 'I-Actor', 'B-InstrumentType', 'I-InstrumentType', 'B-Objective', 'I-Objective', 'B-Resource', 'I-Resource', 'B-Time', 'I-Time']
    elif mode == "mhead":
        label_list = ["O","B","I"]
        head_lst = ["Actor", "InstrumentType", "Objective", "Resource", "Time"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_addr = f"{model_save_addr}/{model_name.split('/')[-1]}_{r}"
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    if mode == "sghead":
        test_dataset = SgheadDataset(dataset_dict["train"], tokenizer, label2id)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: sghead_collate(b, tokenizer.pad_token_id)
        )
        model = AutoModelForTokenClassification.from_pretrained(model_addr).to(dev)
        metrics, preds, reals = sghead_evaluate_model(model, test_loader, dev, id2label, return_rnp=True)
    elif mode == "mhead":
        test_dataset = MheadDataset(dataset_dict["train"], tokenizer, label2id)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: mhead_collate(b, tokenizer.pad_token_id)
        )
        model = MheadTokenClassifier(model_name, head_lst, dropout = dropout).to(dev)
        model.load_state_dict(torch.load(model_addr+"/model.pt", weights_only=True))
        model.eval()
        metrics, preds, reals = mhead_evaluate_model(model, test_loader, dev, id2label, return_rnp=True)
    return metrics, preds, reals

############################## SGHEAD SEQEVAL ##############################

def sghead_getpreds(model_name, label_list, model_save_addr, dsdct_dir, r):
    '''
    Helper function for get_sghead_seqeval
    Uses a saved finetuned model to make predictions on the test split of the datasetdict used to train it
    Returns predictions, real labels, and (for error checking) token_ids
    
    :param model_name: name of base model (huggingface name)
    :param label_list: list of bio labels in integer order
    :param model_save_addr: address to the directory where the models are saved
    :param dsdct_dir: address to the directory where the datasetsdicts are saved
    :param r: which r to use for dataset and model retrieval
    '''
    device = torch.device("cuda")
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    model_tt = AutoModelForTokenClassification.from_pretrained(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_test = dataset_dict["test"]#.map(sghead_tokenize_and_align_labels, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    # setting now before torch conversion, saving for later
    all_inputids = [tokenized_test["input_ids"][i] for i in range(len(tokenized_test["input_ids"]))]
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # set up data collator to work with dataloader in padding inputs to uniform size
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(tokenized_test, batch_size=16, collate_fn=data_collator, shuffle=False)
    all_preds = []
    all_labels = []
    model_tt.eval() # disable dropout bc we're just doing inference
    with torch.no_grad(): # also bc we're just doing inference
        for batch in dataloader:
            # collator returns tensors already padded to max in batch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels")
            # get predictions
            outputs = model_tt(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            # for item in batch, get list of preds and list of labels
            for pred_row, label_row in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                # for (pred,label) in list of preds and labels for that item in batch
                # if the label says it shouldnt be ignored (-100)
                # add the pred to the pred list and the label to the label list
                tp = [label_list[p] for (p, l) in zip(pred_row, label_row) if l != -100]
                tl = [label_list[l] for (p, l) in zip(pred_row, label_row) if l != -100]
                all_preds.append(tp)
                all_labels.append(tl)
    return all_preds, all_labels, all_inputids

def seqeval_for_sghead(predictions, labels):
    '''
    Helper function for get_sghead_seqeval
    Generates results dictionary for provided preds and labels for seqeval on sghead model
    
    :param predictions: list of lists of predictions of labels for each test entry
    :param labels: list of lists of labels for each test entry
    '''
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
    '''
    Loads sghead model to get predictions and labels, then uses seqeval to evaluate results
    Saves predictions, labels, input_ids to one file, then saves results dict to another
    
    :param model_name: name of base model (huggingface name)
    :param label_list: list of bio labels in integer order
    :param model_save_addr: address to the directory where the models are saved
    :param dsdct_dir: address to the directory where the datasetsdicts are saved
    :param r: which r to use for dataset and model retrieval
    :param results_dir: directory where to save resutls files
    '''
    predictions, labels, input_ids = sghead_getpreds(model_name, label_list, model_save_addr, dsdct_dir, r)
    with open(f"{results_dir}/seqeval_{model_name.split('/')[-1]}_{r}_pandr.json", "w", encoding="utf-8") as f:
        json.dump({
            "pred": predictions,
            "real": labels,
            "input_ids": input_ids
        }, f, indent=4)
    results_dict = seqeval_for_sghead(predictions, labels)
    with open(f"{results_dir}/seqeval_{model_name.split('/')[-1]}_{r}_results.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

############################## MHEAD TOKEN F1 ##############################

def mhead_getpreds(model_name, model_save_addr, dsdct_dir, r):
    '''
    Helper function for get_mhead_tokf1
    Uses a saved finetuned model to make predictions on the test split of the datasetdict used to train it
    Returns predictions, real labels, and (for error checking) token_ids
    
    :param model_name: name of base model (huggingface name)
    :param model_save_addr: address to the directory where the models are saved
    :param dsdct_dir: address to the directory where the datasetsdicts are saved
    :param r: which r to use for dataset and model retrieval
    '''
    label_list = ["O", "B", "I"]
    label_cols = [
        "labels_Actor",
        "labels_InstrumentType",
        "labels_Objective",
        "labels_Resource",
        "labels_Time"
    ]
    device = torch.device("cuda")
    # prepare dataset
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_test = dataset_dict["test"]#.map(mhead_tokenize_and_align_labels, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    # setting now before torch conversion, saving for later
    all_inputids = [tokenized_test["input_ids"][i] for i in range(len(tokenized_test["input_ids"]))]
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask"]+label_cols)
    # set up data collator to work with dataloader in padding inputs to uniform size
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(tokenized_test, batch_size=16, collate_fn=data_collator, shuffle=False)
    # dictionaries where we're storing the predictions and labels for each head
    all_preds = {name.replace("labels_",""): [] for name in label_cols}
    all_labels = {name.replace("labels_",""): [] for name in label_cols}
    # model setup
    config = 0#MultiHeadTokenConfig.from_pretrained(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}")
    model_tt = 0#MultiHeadTokClass(config).to(device)
    model_tt.eval() # disable dropout bc we're just doing inference
    with torch.no_grad(): # also bc we're just doing inference
        for batch in dataloader:
            # collator returns tensors already padded to max in batch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {name: batch.get(name) for name in label_cols}
            # get predictions
            outputs = model_tt(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            for head in list(all_preds):
                preds = torch.argmax(logits[head], dim=-1)
                # for item in batch, get list of preds and list of labels
                for pred_row, label_row in zip(preds.cpu().tolist(), labels["labels_"+head].cpu().tolist()):
                    # for (pred,label) in list of preds and labels for that item in batch
                    # if the label says it shouldnt be ignored (-100)
                    # add the pred to the pred list and the label to the label list
                    tp = [label_list[p] for (p, l) in zip(pred_row, label_row) if l != -100]
                    tl = [label_list[l] for (p, l) in zip(pred_row, label_row) if l != -100]
                    all_preds[head].append(tp)
                    all_labels[head].append(tl)
    return all_preds, all_labels, all_inputids

def tokf1_for_mhead(predictions_dct, labels_dct):
    '''
    Helper function for get_mhead_tokf1
    Generates results dictionary for provided preds and labels for tokf1 on mhead model
    
    :param predictions: dict of lists of lists of predictions of labels for each test entry for each featurename
    :param labels: dict of lists of lists of labels for each test entry for each featurename
    '''
    results = {}
    # for each head
    for head_name, preds_lists in predictions_dct.items():
        labels_lists = labels_dct[head_name] # size (batch, seq_len)
        labels_flat = [lbl for lbl_lst in labels_lists for lbl in lbl_lst]
        preds_flat = [pred for pred_lst in preds_lists for pred in pred_lst]
        # micro F1
        f1 = f1_score(labels_flat, preds_flat, average='micro')
        clsf_rpt = classification_report(labels_flat, preds_flat, output_dict=True)
        results[f"{head_name}"] = clsf_rpt
        results[f"{head_name}"]["micro_f1"] = f1
    return results

def seqeval_for_mhead(predictions_dct, labels_dct):
    '''
    Generates results dictionary for provided preds and labels for seqeval on mhead model
    
    :param predictions: dict of lists of lists of predictions of labels for each test entry for each featurename
    :param labels: dict of lists of lists of labels for each test entry for each featurename
    '''
    results_dict = {}
    for ftr in ["Actor", "InstrumentType", "Objective", "Resource", "Time"]:
        results_dict[ftr] = {'precision':[], "recall":[], "f1":[], "number":[]}
    # for each head
    for head_name, preds_lists in predictions_dct.items():
        labels_lists = labels_dct[head_name] # size (batch, seq_len)
        seqeval = evaluate.load("seqeval")
        temp = seqeval.compute(predictions=preds_lists, references=labels_lists)
        print(temp)
        for k in list(temp):
            if k[:4]=="over":
                pass
            else:
                for mtr in list(temp[k]):
                    results_dict[head_name][mtr].append(float(temp[k][mtr]))
    return results_dict

def get_mhead_tokf1(model_name, model_save_addr, dsdct_dir, r, results_dir):
    '''
    Loads sghead model to get predictions and labels, then uses tokf1 to evaluate results
    Saves predictions, labels, input_ids to one file, then saves results dict to another
    
    :param model_name: name of base model (huggingface name)
    :param model_save_addr: address to the directory where the models are saved
    :param dsdct_dir: address to the directory where the datasetsdicts are saved
    :param r: which r to use for dataset and model retrieval
    :param results_dir: directory where to save resutls files
    '''
    predictions, labels, input_ids = mhead_getpreds(model_name, model_save_addr, dsdct_dir, r)
    with open(f"{results_dir}/tokf1_{model_name.split('/')[-1]}_{r}_pandr.json", "w", encoding="utf-8") as f:
        json.dump({
            "pred": predictions,
            "real": labels,
            "input_ids": input_ids
        }, f, indent=4)
    print("Got predictions, calculating results.")
    results_dict = tokf1_for_mhead(predictions, labels)
    with open(f"{results_dir}/tokf1_{model_name.split('/')[-1]}_{r}_results.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    print("Saved results.")

def get_mhead_seqeval(model_name, r, results_dir):
    '''
    Loads sghead model to get predictions and labels, then uses tokf1 to evaluate results
    Saves predictions, labels, input_ids to one file, then saves results dict to another
    
    :param model_name: name of base model (huggingface name)
    :param model_save_addr: address to the directory where the models are saved
    :param dsdct_dir: address to the directory where the datasetsdicts are saved
    :param r: which r to use for dataset and model retrieval
    :param results_dir: directory where to save resutls files
    '''
    with open(f"{results_dir}/tokf1_{model_name.split('/')[-1]}_{r}_pandr.json", "r", encoding="utf-8") as f:
        pandr = json.load(f)
    print("Got predictions, calculating results.")
    results_dict = seqeval_for_mhead(pandr['pred'], pandr['real'])
    with open(f"{results_dir}/seqeval_{model_name.split('/')[-1]}_{r}_results.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    print("Saved results.")

############################## Post-processing of results ##############################

################## sghead seqeval #############################

def visualize_run_sghead_seqeval_results(mode, eval, model_name, r, results_dir, idas=[0]):
    '''
    Currently just for sghead and seqeval
    For each ida (test set entity integer), for each token in the test set entity,
    prints the token next to the prediction next to the label
    For error evaluation
    
    :param mode: sghead or mhead
    :param eval: seqeval or token micro f1
    :param model_name: huggingface model name
    :param r: which model run to look at
    :param results_dir: directory where results files are saved
    :param idas: list of test set entity integers to examine
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(f"{results_dir}/{mode}/{eval}_{model_name.split('/')[-1]}_{r}_pandr.json", "r", encoding="utf-8") as f:
        pandr = json.load(f)
    for ida in idas:
        tokens = tokenizer.convert_ids_to_tokens(pandr['input_ids'][ida])[1:-1]
        print(len(tokens),len(pandr['pred'][ida]), len(pandr['real'][ida]))
        for idx, tok in enumerate(tokens):
            print(f"{idx:03} | {tok:16} | "
                f"P:{pandr['pred'][ida][idx]:16}  "
                f"R:{pandr['real'][ida][idx]:16}  ")

def consol_sghead_seqeval_results(model_names=["microsoft/deberta-v3-base", "dslim/bert-base-NER-uncased"], r_vals=[0,1], results_dir="./"):
    '''
    Given the list of model names and r values, reads the results files and consolidates into a single results file
    
    :param model_names: list of model names whose results it will examine
    :param r_vals: list of r values for those model runs which it will examine
    :param results_dir: directory address where the results files are stored
    '''
    results_dict = {name:{} for name in model_names}
    for model_name in list(results_dict):
        results_dict[model_name]["Overall"] = {"precision":[], "recall":[], "f1":[], "accuracy":[]}
        for ftr in ["Actor", "InstrumentType", "Objective", "Resource", "Time"]:
            results_dict[model_name][ftr] = {"precision":[], "recall":[], "f1":[], "number":[]}
    for model_name in model_names:
        for r in r_vals:
            with open(f"{results_dir}/seqeval_{model_name.split('/')[-1]}_{r}_results.json","r", encoding="utf-8") as f:
                results = json.load(f)
            for k in list(results):
                if k[:4]=="over":
                    x, metric = k.split("_")
                    results_dict[model_name]['Overall'][metric].append(float(results[k][0]))
                else:
                    for mtr in list(results[k]):
                        results_dict[model_name][k][mtr].append(float(results[k][mtr][0]))
    with open(f"{results_dir}/seqeval_results.json","w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    return results_dict

############################# mhead tokf1

def visualize_run_mhead_tokf1_results(mode, eval, model_name, r, results_dir, idas=[0]):
    '''
    Currently just for sghead and seqeval
    For each ida (test set entity integer), for each token in the test set entity,
    prints the token next to the prediction next to the label
    For error evaluation
    
    :param mode: sghead or mhead
    :param eval: seqeval or token micro f1
    :param model_name: huggingface model name
    :param r: which model run to look at
    :param results_dir: directory where results files are saved
    :param idas: list of test set entity integers to examine
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(f"{results_dir}/{mode}/{eval}_{model_name.split('/')[-1]}_{r}_pandr.json", "r", encoding="utf-8") as f:
        pandr = json.load(f)
    for ida in idas:
        tokens = tokenizer.convert_ids_to_tokens(pandr['input_ids'][ida])[1:-1]
        #print(len(tokens),len(pandr['pred'][head][ida]), len(pandr['real'][head][ida]))
        for idx, tok in enumerate(tokens):
            try:
                print(f"{idx:03} | {tok:16} | "
                    f"| {pandr['real']['Actor'][ida][idx]:1}  "
                    f"{pandr['pred']['Actor'][ida][idx]:1}  "
                    f"| {pandr['real']['InstrumentType'][ida][idx]:1}  "
                    f"{pandr['pred']['InstrumentType'][ida][idx]:1}  "
                    f"| {pandr['real']['Objective'][ida][idx]:1}  "
                    f"{pandr['pred']['Objective'][ida][idx]:1}  "
                    f"| {pandr['real']['Resource'][ida][idx]:1}  "
                    f"{pandr['pred']['Resource'][ida][idx]:1}  "
                    f"| {pandr['real']['Time'][ida][idx]:1}  "
                    f"{pandr['pred']['Time'][ida][idx]:1}  ")
            except IndexError:
                pass

def consol_mhead_tokf1_results(model_names=["microsoft/deberta-v3-base", "dslim/bert-base-NER-uncased"], r_vals=[0,1], results_dir="./"):
    '''
    Given the list of model names and r values, reads the results files and consolidates into a single results file
    
    :param model_names: list of model names whose results it will examine
    :param r_vals: list of r values for those model runs which it will examine
    :param results_dir: directory address where the results files are stored
    '''
    results_dict = {name:{} for name in model_names}
    for model_name in list(results_dict):
        for ftr in ["Actor", "InstrumentType", "Objective", "Resource", "Time"]:
            results_dict[model_name][ftr] = {"micro_f1":[]}
    for model_name in model_names:
        for r in r_vals:
            with open(f"{results_dir}/tokf1_{model_name.split('/')[-1]}_{r}_results.json","r", encoding="utf-8") as f:
                results = json.load(f)
            for ftrname in list(results):
                results_dict[model_name][ftrname]["micro_f1"].append(float(results[ftrname]["micro_f1"]))
    with open(f"{results_dir}/tokf1_results.json","w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    return results_dict

def consol_mhead_seqeval_results(model_names=["microsoft/deberta-v3-base", "dslim/bert-base-NER-uncased"], r_vals=[0,1], results_dir="./"):
    '''
    Given the list of model names and r values, reads the results files and consolidates into a single results file
    
    :param model_names: list of model names whose results it will examine
    :param r_vals: list of r values for those model runs which it will examine
    :param results_dir: directory address where the results files are stored
    '''
    results_dict = {name:{} for name in model_names}
    for model_name in list(results_dict):
        for ftr in ["Actor", "InstrumentType", "Objective", "Resource", "Time"]:
            results_dict[model_name][ftr] = {"precision":[], "recall":[], "f1":[], "number":[]}
    for model_name in model_names:
        for r in r_vals:
            with open(f"{results_dir}/seqeval_{model_name.split('/')[-1]}_{r}_results.json","r", encoding="utf-8") as f:
                results = json.load(f)
            for k in list(results):
                if k[:4]=="over":
                    pass
                else:
                    for mtr in list(results[k]):
                        results_dict[model_name][k][mtr].append(float(results[k][mtr][0]))
    with open(f"{results_dir}/seqeval_results.json","w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    return results_dict

def shortestvis_tokf1(results_dict):
    '''
    Given *consolidated* results dictionary, prints only the rounded, averaged values across runs
    
    :param results_dict: results dictionary passed from consol_sghead_seqeval_results or loaded from file
    '''
    for m in list(results_dict):
        print(f"\n{m}")
        for res in list(results_dict[m]):
            print(f"\n{res}")
            df = pd.DataFrame(results_dict[m][res])
            df.loc['mean'] = df.mean()            
            df.loc['sd'] = df.std()
            print(round(df.loc['mean']*100,2))
            print(round(df.loc['sd']*100,2))

########################### other

def df_vis_consol_sghead_seqeval(results_dict):
    '''
    Given *consolidated* results dictionary, prints in dataframe format
    
    :param results_dict: results dictionary passed from consol_sghead_seqeval_results or loaded from file
    '''
    for m in list(results_dict):
        print(f"\n{m}")
        for res in list(results_dict[m]):
            print(f"\n{res}")
            df = pd.DataFrame(results_dict[m][res])
            df.loc['mean'] = df.mean()
            print(df)

def shortestvis(results_dict):
    '''
    Given *consolidated* results dictionary, prints only the rounded, averaged values across runs
    
    :param results_dict: results dictionary passed from consol_sghead_seqeval_results or loaded from file
    '''
    for m in list(results_dict):
        print(f"\n{m}")
        for res in list(results_dict[m]):
            print(f"\n{res}")
            df = pd.DataFrame(results_dict[m][res])
            df.loc['mean'] = df.mean()
            df.loc['sd'] = df.std()
            print(round(df.loc['mean']*100,2))
            print(round(df.loc['sd']*100,2))

########
# main #
########

def main():
    cwd = os.getcwd()
    results_dir = cwd+"/../results"
    sghead_dsdcts_dir = cwd+"/../inputs/sghead_dsdcts"
    sghead_models_dir = cwd+"/../models/sghead"
    mhead_dsdcts_dir = cwd+"/../inputs/mhead_dsdcts"
    mhead_models_dir = cwd+"/../models/mhead"
    label_list = ['O', 'B-Actor', 'I-Actor', 'B-InstrumentType', 'I-InstrumentType', 'B-Objective', 'I-Objective', 'B-Resource', 'I-Resource', 'B-Time', 'I-Time']

    model_name = "FacebookAI/xlm-roberta-base"
    r = 1
    metrics, preds, reals = evaluate_model("sghead", model_name, sghead_models_dir, sghead_dsdcts_dir, r, batch_size=16)
    print("\n", metrics)
    for i in range(20):
        print(preds[0][i], reals[0][i])
    metrics, preds, reals = evaluate_model("mhead", model_name, mhead_models_dir, mhead_dsdcts_dir, r)
    print("\n", metrics)
    for head in list(preds):
        print(head)
        for i in range(20):
            print(preds[head][0][0][i], reals[head][0][0][i])
    #print(metrics)
    #cult_metr = extract_results(metrics, "sghead", "seqeval")
    metrics = {'Actor': {'_': {'precision': 0.8, 'recall': 0.8, 'f1': 0.8000000000000002, 'number': 120}, 'overall_precision': 0.8, 'overall_recall': 0.8, 'overall_f1': 0.8000000000000002, 'overall_accuracy': 0.986378442404501}, 'InstrumentType': {'_': {'precision': 0.5306122448979592, 'recall': 0.2549019607843137, 'f1': 0.3443708609271523, 'number': 102}, 'overall_precision': 0.5306122448979592, 'overall_recall': 0.2549019607843137, 'overall_f1': 0.3443708609271523, 'overall_accuracy': 0.9452176488007107}, 'Objective': {'_': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'number': 15}, 'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_accuracy': 0.9700917974533609}, 'Resource': {'_': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'number': 27}, 'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_accuracy': 0.9878590464909683}, 'Time': {'_': {'precision': 0.07692307692307693, 'recall': 0.0625, 'f1': 0.06896551724137931, 'number': 16}, 'overall_precision': 0.07692307692307693, 'overall_recall': 0.0625, 'overall_f1': 0.06896551724137931, 'overall_accuracy': 0.9813443885105123}, 'avg_eval_loss': 0.452874297897021}
    #cult_metr = extract_results(metrics, "mhead", "seqeval")
    print("\n")
    #print(cult_metr)
    '''
    with open(cwd+"/../inputs/sghead_ds/label_mapping.json", "r", encoding="utf-8") as f:
        label_list = json.load(f)
    #
    model_name = "FacebookAI/xlm-roberta-base"
    r = 3
    #get_sghead_seqeval(model_name, label_list, sghead_models_dir, sghead_dsdcts_dir, r, results_dir+"/sghead")
    get_mhead_tokf1(model_name, mhead_models_dir, mhead_dsdcts_dir, r, results_dir+"/mhead")
    #visualize_run_mhead_tokf1_results("mhead", "tokf1", model_name, r, results_dir, idas=[0,1,2])
    
    for model_name in ["microsoft/deberta-v3-base", "dslim/bert-base-NER-uncased", "FacebookAI/xlm-roberta-base"]:
        for r in [3,4,5]:#[0,1,2]:
            print(f"\n{model_name} {r}")
            #get_sghead_seqeval(model_name, label_list, sghead_models_dir, sghead_dsdcts_dir, r, results_dir+"/sghead")
            get_mhead_tokf1(model_name, mhead_models_dir, mhead_dsdcts_dir, r, results_dir+"/mhead")
            get_mhead_seqeval(model_name, r, results_dir+"/mhead")
    
    #results_dict = consol_sghead_seqeval_results(model_names=["microsoft/deberta-v3-base", "dslim/bert-base-NER-uncased", "FacebookAI/xlm-roberta-base"], r_vals=[0,1,2,3,4,5], results_dir=results_dir+"/sghead")
    results_dict = consol_mhead_tokf1_results(model_names=["microsoft/deberta-v3-base", "dslim/bert-base-NER-uncased", "FacebookAI/xlm-roberta-base"], r_vals=[0,1,2,3,4,5], results_dir=results_dir+"/mhead")
    #results_dict = consol_mhead_seqeval_results(model_names=["microsoft/deberta-v3-base", "dslim/bert-base-NER-uncased", "FacebookAI/xlm-roberta-base"], r_vals=[0,1,2], results_dir=results_dir+"/mhead")
    
    #with open(f"{results_dir}/sghead/seqeval_results.json","r", encoding="utf-8") as f:
    with open(f"{results_dir}/mhead/tokf1_results.json","r", encoding="utf-8") as f:
        results_dict = json.load(f)
    
    #df_vis_consol_sghead_seqeval(results_dict)

    #shortestvis(results_dict)
    shortestvis_tokf1(results_dict)'''
    ''''''
    # postprocessing
    #visualize_run_results("sghead", "seqeval", model_name, r, results_dir, [0,2])

if __name__=="__main__":
    main()
