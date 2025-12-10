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

#############
# functions #
#############

############################## SGHEAD SEQEVAL ##############################

def sghead_getpreds(model_name, label_list, model_save_addr, dsdct_dir, r):
    device = torch.device("cuda")
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    model_tt = AutoModelForTokenClassification.from_pretrained(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}").to(device)
    ############################################
    # initialize tokenization of dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_and_align_labels(examples):
        # fxn from https://huggingface.co/docs/transformers/en/tasks/token_classification
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    tokenized_test = dataset_dict["test"].map(tokenize_and_align_labels, batched=True)
    all_inputids = [tokenized_test["input_ids"][i] for i in range(len(tokenized_test["input_ids"]))]
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(tokenized_test, batch_size=16, collate_fn=data_collator, shuffle=False)
    all_preds = []
    all_labels = []
    model_tt.eval()
    with torch.no_grad():
        for batch in dataloader:
            # collator returns tensors already padded to max in batch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)
            outputs = model_tt(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            for pred_row, label_row in zip(preds.cpu().tolist(), (labels.cpu().tolist() if labels is not None else [None]*preds.size(0))):
                if label_row is None:
                    all_preds.append([label_list[p] for p in pred_row])
                    all_labels.append(None)
                else:
                    tp = [label_list[p] for (p, l) in zip(pred_row, label_row) if l != -100]
                    tl = [label_list[l] for (p, l) in zip(pred_row, label_row) if l != -100]
                    all_preds.append(tp)
                    all_labels.append(tl)
    #print(len(all_inputids[0]), len(all_preds[0]), len(all_labels[0]))
    return all_preds, all_labels, all_inputids

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

############################## Post-processing of results ##############################

def visualize_run_sghead_seqeval_results(mode, eval, model_name, r, results_dir, idas=[0]):
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

def df_vis_consol_sghead_seqeval(results_dict):
    for m in list(results_dict):
        print(f"\n{m}")
        for res in list(results_dict[m]):
            print(f"\n{res}")
            df = pd.DataFrame(results_dict[m][res])
            df.loc['mean'] = df.mean()
            print(df)

def shortestvis(results_dict):
    for m in list(results_dict):
        print(f"\n{m}")
        for res in list(results_dict[m]):
            print(f"\n{res}")
            df = pd.DataFrame(results_dict[m][res])
            df.loc['mean'] = df.mean()
            print(round(df.loc['mean']*100,2))

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
    #model_name = "microsoft/deberta-v3-base"
    #r = 3
    '''
    for model_name in ["microsoft/deberta-v3-base", "dslim/bert-base-NER-uncased", "FacebookAI/xlm-roberta-base"]:
        for r in [0,1,2]:
            print(f"\n{model_name} {r}")
            get_sghead_seqeval(model_name, label_list, sghead_models_dir, sghead_dsdcts_dir, r, results_dir+"/sghead")

    results_dict = consol_sghead_seqeval_results(model_names=["microsoft/deberta-v3-base", "dslim/bert-base-NER-uncased", "FacebookAI/xlm-roberta-base"], r_vals=[0,1,2], results_dir=results_dir+"/sghead")
    '''
    with open(f"{results_dir}/sghead/seqeval_results.json","r", encoding="utf-8") as f:
        results_dict = json.load(f)
    
    #df_vis_consol_sghead_seqeval(results_dict)
    shortestvis(results_dict)
    # postprocessing
    #visualize_run_results("sghead", "seqeval", model_name, r, results_dir, [0,2])

if __name__=="__main__":
    main()