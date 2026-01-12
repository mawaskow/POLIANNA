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

#############
# functions #
#############

def sghead_tokenize_and_align_labels(tokens, ner_tags, tokenizer, label2id, max_length=512):
    tokenized_inputs = tokenizer(tokens, truncation=True, is_split_into_words=True, max_length=max_length, return_tensors=None)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label2id[ner_tags[word_idx]])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

class SgheadDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, label2id, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]
        encoding = sghead_tokenize_and_align_labels(
            tokens,
            ner_tags,
            self.tokenizer,
            self.label2id,
            self.max_length
        )
        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(encoding["labels"])
        }

def sghead_collate(batch, pad_token_id):
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

def finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params = None):
    if not params:
        params = {
            "num_epochs": 10,
            "lr": 3e-5,
            "weight_decay": 0.01,
            "batch_size":16
        }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    train_dataset = SgheadDataset(dataset_dict["train"], tokenizer)
    dev_dataset = SgheadDataset(dataset_dict["dev"], tokenizer)
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
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=(model_name == "dslim/bert-base-NER-uncased")
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    num_training_steps = params["num_epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    model.train()
    for epoch in range(params["num_epochs"]):
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train loss: {avg_loss:.4f}")
    seqeval = evaluate.load("seqeval")
    def evaluate_model(model, dataloader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                labels = batch["labels"]
                batch = {k: v.to(dev) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = labels.numpy()
                for preds, labs in zip(predictions, labels):
                    true_preds = []
                    true_labs = []
                    for p, l in zip(preds, labs):
                        if l != -100:
                            true_preds.append(id2label[p])
                            true_labs.append(id2label[l])
                    all_preds.append(true_preds)
                    all_labels.append(true_labs)
        return seqeval.compute(predictions=all_preds, references=all_labels)
    metrics = evaluate_model(model, dev_loader)
    save_path = f"{model_save_addr}/{model_name.split('/')[-1]}_{r}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
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
    
    model_name = "microsoft/deberta-v3-base"
    r = 0
    finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r)
    ''''''
    '''
    ########### loop mode ###########
    #["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]
    st = time.time()
    for model_name in ["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]:
        md_st = time.time()
        for r in [0,1,2]:
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