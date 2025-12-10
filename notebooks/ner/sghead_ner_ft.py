'''
Normal token classification model training 
'''
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import DatasetDict
import evaluate
import numpy as np
import json
import subprocess
import time

#############
# functions #
#############

def finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r):
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
    # tokenize dataset and initialize data collator
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    tokenized_dsdct = dataset_dict.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # get labels
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    # setup metrics calculation
    seqeval = evaluate.load("seqeval")
    def compute_metrics(p):
        # fxn from https://huggingface.co/docs/transformers/en/tasks/token_classification
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    # load model
    igmms = model_name == "dslim/bert-base-NER-uncased"
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=igmms
    )
    print(model_name, "loaded.")
    # define trainer
    training_args = TrainingArguments(
        output_dir=model_name.split("/")[-1],
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dsdct["train"],
        eval_dataset=tokenized_dsdct["dev"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    # train
    print("Training begin")
    trainer.train()
    trainer.save_model(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}")
    del model
    del trainer
    del tokenizer
    print(f"Trained {model_name} r{r}")

########
# main #
########

def main():
    cwd = os.getcwd()
    model_save_addr = cwd+"/../models/sghead"
    dsdct_dir = cwd+"/../inputs/sghead_dsdcts"
    with open(cwd+"/../inputs/sghead_ds/label_mapping.json", "r", encoding="utf-8") as f:
        label_list = json.load(f)
    ########### one-off ###########
    #model_name = "microsoft/deberta-v3-base"
    #r = 3
    #finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r)
    ########### loop mode ###########
    st = time.time()
    for model_name in ["microsoft/deberta-v3-base"]:
        md_st = time.time()
        for r in [3]:
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

if __name__=="__main__":
    main()