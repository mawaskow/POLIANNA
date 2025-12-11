'''
A custom token classification model with five separate classifier heads for five separate feature types.
'''
import os
import numpy as np
from collections import Counter
from transformers import AutoModel, AutoConfig, PreTrainedTokenizerBase, PretrainedConfig, PreTrainedModel
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import DatasetDict
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List
from sklearn.metrics import f1_score
import subprocess
import gc
import time

#########################
# classes and functions #
#########################

class MultiHeadDataCollator:
    '''
    New data collator class, making our own since it's less complicated than trying to adapt hf collators
    Using PreTrainedTokenizerBase i.e. whatever pretrained tokenizer we have from tokenize_and_align_labels
    And using pad_sequence
    '''
    def __init__(self, tokenizer: PreTrainedTokenizerBase, label_columns: List[str], padding=True, max_length=None):
        #initializing the essentials
        self.tokenizer = tokenizer
        self.label_columns = label_columns
        self.padding = padding
        self.max_length = max_length
    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        #attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features] # something isnt working
        attention_mask = [
            torch.tensor(f.get("attention_mask", [1]*len(f["input_ids"])), dtype=torch.long)
            for f in features
        ]
        # padding inputids and attnmask
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        # batch
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        # padding labels
        for col in self.label_columns:
            #label_lists = [torch.tensor(f[col], dtype=torch.long) for f in features] # something isnt working here either
            label_lists = [
                torch.tensor(f.get(col, [-100]*len(f["input_ids"])), dtype=torch.long)
                for f in features
            ]
            labels_padded = pad_sequence(label_lists, batch_first=True, padding_value=-100)
            # then finally adding to batch
            batch[col] = labels_padded
        return batch

def get_weights(label_cols, tokenized_dsdct):
    '''
    Takes the tokenized datasetdict and calcuates inverse frequency weights for 
    A) the BIO labels in each feature category
    B) the feature heads based on B and I token counts
    
    :param label_cols: list of label list keys in dataset
    :param tokenized_dsdct: tokenized datasetdict
    Returns dictionary of weights
    '''
    # lets create class (BIO) weights for each feature type
    num_classes = 3
    class_weights = {}
    class_weights_norm = {}
    for lname in label_cols:
        name = lname.replace("labels_", "")
        labels = np.concatenate([np.array(l) for l in tokenized_dsdct['train'][lname]])
        labels = labels[labels != -100]  # remove padding
        counter = Counter(labels)
        # inverse frequency weighting
        weights=[0]*num_classes
        for i in range(num_classes):
            count = counter.get(i, 0)
            if count == 0: #handle no-shows so no zer-os (no dividing by zeros that is)
                weights[i] = 1.0
            else:
                weights[i] = len(labels) / (num_classes * count)
        class_weights[name] = torch.tensor(weights, dtype=torch.float)
        # then normalize
        w = torch.tensor(weights, dtype=torch.float)
        w = w / w.mean()
        class_weights_norm[name] = w
    # weights for each head
    head_counts = {}
    for lname in label_cols:
        name = lname.replace("labels_", "")
        # concatenate all labels and remove -100s
        labels = np.concatenate([np.array(l) for l in tokenized_dsdct['train'][lname]])
        labels = labels[labels != -100]
        labels = labels[labels != 0]
        head_counts[name] = len(labels) # only tokens B or I
    total_tokens = sum(head_counts.values())
    head_weights = {head: total_tokens / (len(head_counts) * count) for head, count in head_counts.items()}
    # normalize
    w = torch.tensor(list(head_weights.values()), dtype=torch.float)
    w = w / w.mean()
    head_weights_norm = {head: w[i] for i, head in enumerate(head_weights.keys())}
    return {"class_weights": class_weights, "class_weights_norm": class_weights_norm, "head_weights": head_weights, "head_weights_norm": head_weights_norm}

# this is the output of get_weights(label_cols, tokenized_dsdct) for our original <512tokens dataset
WEIGHTS ={'class_weights': {'Actor': torch.tensor([ 0.3560,  9.8114, 11.2143]),
  'InstrumentType': torch.tensor([ 0.3483, 16.4053, 14.7887]),
  'Objective': torch.tensor([ 0.3513, 53.5521,  7.4079]),
  'Resource': torch.tensor([ 0.3367, 92.7964, 53.0844]),
  'Time': torch.tensor([ 0.3412, 59.8834, 19.0240])},
 'class_weights_norm': {'Actor': torch.tensor([0.0500, 1.3766, 1.5734]),
  'InstrumentType': torch.tensor([0.0331, 1.5603, 1.4066]),
  'Objective': torch.tensor([0.0172, 2.6203, 0.3625]),
  'Resource': torch.tensor([0.0069, 1.9039, 1.0892]),
  'Time': torch.tensor([0.0129, 2.2669, 0.7202])},
 'head_weights': {'Actor': 0.5988807576409815,
  'InstrumentType': 0.8900831733845169,
  'Objective': 0.7447537473233404,
  'Resource': 3.8644444444444446,
  'Time': 1.6522565320665084},
 'head_weights_norm': {'Actor': torch.tensor(0.3864),
  'InstrumentType': torch.tensor(0.5742),
  'Objective': torch.tensor(0.4805),
  'Resource': torch.tensor(2.4931),
  'Time': torch.tensor(1.0659)}}

class MultiHeadTokenConfig(PretrainedConfig):
    '''
    Apparently if we want to load our custom finetuned model using from_pretrained for inference later
    then we have to make a special config for it so hf can load it :/
    '''
    model_type = "multihead"
    def __init__(
            self,
            base_model_name="microsoft/deberta-v3-base",
            n_labels=3,
            heads=None,
            hidden_size=None,
            id2label=None,
            label2id=None,
            **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.n_labels = n_labels
        self.heads = heads or ["Actor", "InstrumentType", "Objective", "Resource", "Time"]
        self.hidden_size = hidden_size 
        self.id2label = id2label or {0: "O", 1: "B", 2: "I"}
        self.label2id = label2id or {v: k for k, v in self.id2label.items()}

class MultiHeadTokClass(PreTrainedModel):
    config_class = MultiHeadTokenConfig
    def __init__(self, config):
        super().__init__(config)
        self.encoder_config = AutoConfig.from_pretrained(config.base_model_name) # connecting our config info
        self.encoder = AutoModel.from_pretrained(config.base_model_name, config=self.encoder_config)
        hidden_size = self.base_model.config.hidden_size # explicitly adding this bc it triggered an error
        #sep linear head for each feature type classification
        self.classifiers = nn.ModuleDict({
            head: nn.Linear(hidden_size, config.num_labels) for head in config.heads
        })
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        self.init_weights()
    def forward(self, input_ids, attention_mask=None, **labels):
        # batch of inputs encoded by base model
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        # only uses last hidden state... for now
        # will look into averaging/concatenating last few hidden states
        sequence_output = outputs.last_hidden_state
        #sequence_output = self.dropout(sequence_output)
        # passes encoded input sequence to each classifier to get logits
        logits = {name: self.classifiers[name](sequence_output) for name in self.classifiers}
        loss = None
        if labels:
            loss = 0
            # for labels_Feature, tensor(batch_sz,seq_ln)
            for lname, label in labels.items():
                if label is not None:
                    name = lname.replace("labels_", "")
                    # flatten attn mask
                    active_loss = attention_mask.view(-1) == 1
                    # get active logits, flatten to (num_act_tokens, num_classes) 
                    # then apply active loss mask (to both logits and labels)
                    active_logits = logits[name].view(-1, 3)[active_loss]
                    active_labels = label.view(-1)[active_loss]
                    # weighting BIO classes for this feature
                    weight = WEIGHTS['class_weights_norm'][name].to(active_logits.device)
                    loss_fct = nn.CrossEntropyLoss(weight=weight)
                    # computing loss for this head
                    head_loss = loss_fct(active_logits, active_labels)
                    # weight the loss for this head
                    head_loss *= WEIGHTS['head_weights_norm'][name]
                    # sum loss across heads for single update to train simultaneously
                    loss += head_loss
        # has to be TokenClassifierOutput or else Trainer doesn't pick up on evaluation loss
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )

class MultiHeadTrainer(Trainer):
    '''
    Custom trainer that calculates evaluation loss for the mhead model
    '''
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # have to add num_items_in_batch bc it's expected even if it isnt used
        # extract labels for each head from inputs
        labels = {k: inputs.pop(k) for k in list(inputs.keys()) if k.startswith("labels_")}
        # forward pass
        outputs = model(**inputs, **labels)
        # model returns TokenClassifierOutput
        loss = outputs.loss
        if return_outputs: # also added bc even if it isnt used sometimes hf models require it ig? idk
            return loss, outputs
        return loss

def compute_metrics_mhead(p):
    '''
    Custom F1 score metrics calculation for mhead model
    
    :param p: evaluation prediction item contianing predictions and labels
    '''
    prediction_dct, label_dct = p
    lblnames = [i[0] for i in prediction_dct.items()]
    metrics = {}
    # for each head
    for head_name, logits in prediction_dct.items():
        labels = label_dct[lblnames.index(head_name)] # size (batch, seq_len)
        labels_flat = labels.flatten()
        preds_flat = np.argmax(logits, axis=-1).flatten()
        # mask out -100s
        mask = labels_flat != -100
        labels_flat = labels_flat[mask]
        preds_flat = preds_flat[mask]
        # micro F1
        f1 = f1_score(labels_flat, preds_flat, average='micro')
        metrics[f"{head_name}_f1"] = f1
    return metrics

def mhead_tokenize_and_align_labels(examples, tokenizer):
    '''
    adapted for multi-head from https://huggingface.co/docs/transformers/en/tasks/token_classification
    even tho the token lists area already split into words, we need to break them into subwords
    and then ensure that the label sequences still align in the new token sequence
    
    :param examples: datasetdict
    :param tokenizer: tokenizer
    '''
    label_cols = [
        "labels_Actor",
        "labels_InstrumentType",
        "labels_Objective",
        "labels_Resource",
        "labels_Time"
    ]
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding=True, return_attention_mask=True)
    # for each label type/list
    for col in label_cols:
        all_aligned_labels = []
        # loop through this label type's sequence in each sample and realign
        for sample_idx, labels in enumerate(examples[col]):
            word_ids = tokenized_inputs.word_ids(batch_index=sample_idx)
            # smth like [None, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20]
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            all_aligned_labels.append(label_ids)
        tokenized_inputs[col] = all_aligned_labels
    return tokenized_inputs

def finetune_mhead_model(model_name, model_save_addr, dsdct_dir, r):
    '''
    Docstring for finetune_mhead_model
    
    :param model_name: name of hf model to be used as tokenizer and base model for fine-tuning
    :param model_save_addr: directory address where to save the model directories
    :param dsdct_dir: directory address (mhead_dsdcts) where the datasetdictionary directories (dsdct_r{#}) are stored
    :param r: which # run
    '''
    # initialize labels and label fields in the dataset
    label2id = {"O":0, "B":1, "I":2}
    id2label = {0:"O", 1:"B", 2:"I"}
    label_cols = [
        "labels_Actor",
        "labels_InstrumentType",
        "labels_Objective",
        "labels_Resource",
        "labels_Time"
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # load and tokenize datasetdict
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    tokenized_dsdct = dataset_dict.map(mhead_tokenize_and_align_labels, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    data_collator = MultiHeadDataCollator(tokenizer=tokenizer, label_columns=label_cols, max_length=512)
    # initialize config so we can use from_pretrained to load our models in the eval stage
    encoder_config = AutoConfig.from_pretrained(model_name)
    config = MultiHeadTokenConfig(
        base_model_name=model_name,
        num_labels=3,
        heads=["Actor", "InstrumentType", "Objective", "Resource", "Time"],
        hidden_size=encoder_config.hidden_size,
        id2label=id2label,
        label2id=label2id
    )
    model = MultiHeadTokClass(config)
    training_args = TrainingArguments(
        output_dir=model_name.split("/")[-1],
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        label_names=label_cols
    )
    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dsdct["train"],
        eval_dataset=tokenized_dsdct["dev"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_mhead
    )
    # train
    trainer.train()
    #trainer.save_model(cwd+f"/models/{mode}/{model_name.split('/')[-1]}_{r}")
    model.save_pretrained(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}")
    config.save_pretrained(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}")
    del model
    del trainer
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

########
# main #
########

def main():
    cwd = os.getcwd()
    model_save_addr = cwd+"/../models/mhead"
    dsdct_dir = cwd+"/../inputs/mhead_dsdcts"
    ########### one-off ###########
    ''''''
    model_name = "FacebookAI/xlm-roberta-base"
    r = 3
    finetune_mhead_model(model_name, model_save_addr, dsdct_dir, r)
    
    ########### loop mode ###########
    '''
    st = time.time()
    for model_name in ["FacebookAI/xlm-roberta-base"]:
        md_st = time.time()
        for r in [3]:
            print(f"\n--- Starting run {model_name} r{r} ---")
            run_st = time.time()
            subprocess.run([
                "python", "train_mhead.py",
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
