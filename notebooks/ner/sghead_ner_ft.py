'''
Normal token classification model training 
'''
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from datasets import DatasetDict
import evaluate
import numpy as np
import json
import subprocess
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import gc
import tqdm

#############
# functions #
#############

def convert_numpy_torch_to_python(obj):
    '''
    For converting our metrics dict to something json serializable
    '''
    if isinstance(obj, dict):
        return {k: convert_numpy_torch_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_torch_to_python(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

class SgheadDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, label2id, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    def __len__(self):
        return len(self.dataset)
    def tokenize_and_align_labels(self, tokens, ner_tags):
        tokenized_inputs = self.tokenizer(tokens, truncation=True, is_split_into_words=True, max_length=self.max_length, return_tensors=None)
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(self.label2id[ner_tags[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]
        encoded = self.tokenize_and_align_labels(tokens, ner_tags)
        return {
            "input_ids": torch.tensor(encoded["input_ids"]),
            "attention_mask": torch.tensor(encoded["attention_mask"]),
            "labels": torch.tensor(encoded["labels"])
        }

def sghead_collate(batch, pad_token_id):
    '''
    Collate function for our pytorch DataLoader
    '''
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }

def sghead_evaluate_model(model, dataloader, dev, id2label, return_rnp = False):
        seqeval = evaluate.load("seqeval")
        model.eval()
        total_eval_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(dev) for k, v in batch.items()}
                outputs = model(**batch)
                total_eval_loss += outputs.loss.item()
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                for preds, labs in zip(predictions, labels):
                    true_preds = []
                    true_labs = []
                    for p, l in zip(preds, labs):
                        if l != -100:
                            true_preds.append(id2label[p])
                            true_labs.append(id2label[l])
                    all_preds.append(true_preds)
                    all_labels.append(true_labs)
        metrics = seqeval.compute(predictions=all_preds, references=all_labels)
        metrics['avg_eval_loss'] = total_eval_loss / len(dataloader)
        if return_rnp:
            return metrics, all_preds, all_labels
        return metrics

def finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params = None):
    '''
    Docstring for finetune_sghead_model
    
    :param model_name: model to finetune
    :param label_list: list of labels
    :param model_save_addr: where to save model
    :param dsdct_dir: location of datasetdicts
    :param r: which dataset split dict to use
    :param params: hyperparameter dict including num_epochs, lr, weight_decay, batch_size, num_warmup_steps, and patience
    '''
    st = time.time()
    if not params:
        params = {
            "num_epochs": 10,
            "lr": 3e-5,
            "weight_decay": 0.01,
            "batch_size":16,
            "num_warmup_steps":0,
            "patience": 3
        }
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    train_dataset = SgheadDataset(dataset_dict["train"], tokenizer, label2id)
    dev_dataset = SgheadDataset(dataset_dict["dev"], tokenizer, label2id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=lambda b: sghead_collate(b, tokenizer.pad_token_id)
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=lambda b: sghead_collate(b, tokenizer.pad_token_id)
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes= model_name == "dslim/bert-base-NER-uncased"
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    num_training_steps = params["num_epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params["num_warmup_steps"],
        num_training_steps=num_training_steps
    )
    # adding early stopping
    best_eval_loss = float("inf")
    epochs_no_improvement = 0
    model_epoch = 0
    for epoch in range(params["num_epochs"]):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm.tqdm(train_loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            train_loss = outputs.loss
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += train_loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        metrics = sghead_evaluate_model(model, dev_loader, dev, id2label)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {metrics['avg_eval_loss']:.4f} | Precision: {metrics['overall_precision']:.4f} | Recall: {metrics['overall_recall']:.4f} | F1: {metrics['overall_f1']:.4f}")
        if metrics['avg_eval_loss'] < best_eval_loss:
            best_eval_loss = metrics['avg_eval_loss']
            model_epoch = epoch
            epochs_no_improvement = 0
            save_path = f"{model_save_addr}/{model_name.split('/')[-1]}_{r}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= params['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    metrics['epoch_saved'] = model_epoch
    metrics['time_min'] = round((time.time()-st)/60,2)
    print(metrics)
    with open(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}/params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    metrics_clean = convert_numpy_torch_to_python(metrics)
    with open(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_clean, f, ensure_ascii=False, indent=4)
    # cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

########
# main #
########

def main():
    cwd = os.getcwd()
    model_save_addr = cwd+"/../models/sghead"
    dsdct_dir = cwd+"/../inputs/sghead_dsdcts"
    label_list = ['O', 'B-Actor', 'I-Actor', 'B-InstrumentType', 'I-InstrumentType', 'B-Objective', 'I-Objective', 'B-Resource', 'I-Resource', 'B-Time', 'I-Time']
    ########### one-off ###########
    
    model_name = "FacebookAI/xlm-roberta-base"
    r = 0
    params = {
            "num_epochs": 15,
            "lr": 3e-5,
            "weight_decay": 0.01,
            "batch_size":16,
            "num_warmup_steps":0,
            "patience": 3
        }
    finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params)
    ''''''
    '''
    ########### loop mode ###########
    #["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]
    st = time.time()
    for model_name in ["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]:
        md_st = time.time()
        for r in list(range(5)):
            print(f"\n--- Starting run {model_name} r{r} ---")
            run_st = time.time()
            subprocess.run([
                "python", "train_sghead.py",
                model_name,
                str(r),
                model_save_addr,
                dsdct_dir
            ],
                check=True, capture_output=True, text=True)
            print(f"\n--- Finished run {model_name} r{r} ---")
            print(f'\nRun done in {round((time.time()-run_st)/60,2)} min')
            time.sleep(2)
        print(f"\nAll r's of {model_name} done in {round((time.time()-md_st)/60,2)} min")
    print(f'\nAll models and runs done in {round((time.time()-st)/60,2)} min')
    '''

if __name__=="__main__":
    main()