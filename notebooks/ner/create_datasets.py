'''
Creates the datasets. 
*****Currently only uses articles that are fewer than 512 tokens.
*****Currently only looking at Actor, InstrumentType, Objective, Resource, Time
Options are sghead and mhead
sghead: labels are B-class, I-class, O for all classes in one label set
mhead: labels are BIO format in a different label set for each class
'''
import json
import sys
import pandas as pd
import os
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import Dataset as TorchDataset
sys.path.insert(0, '../..') # ensures access to src module

#############
# functions #
#############

def df_loading(pol_dir, col_sel=["Policy","Text","Tokens","Curation"]):
    '''
    Function that loads dataframe from POLIANNA pkl
    :param pol_dir: directory where preprocessed_dataframe.pkl is stored
    :param col_sel: columns of interest in df
    '''
    pol_df = pd.read_pickle(pol_dir+"/preprocessed_dataframe.pkl")[col_sel]
    return pol_df

################################ SGHEAD ################################

def extract_sghead_label_set(df):
    '''
    Helper function for df_to_sghead_ds(df)
    Function that basically just generates a label list that'll stay the same for any sghead run
    but at least with a function we remember how we filtered down to these labels.
    :param df: polianna dataframe
    '''
    labels = set()
    for spancoll in df["Curation"]:
        for spn in spancoll:
            if spn.layer in ["Instrumenttypes", "Policydesigncharacteristics"]:
                ftr = spn.feature
                if ftr not in ["Compliance","Reversibility", "Reference", "InstrumentType_2", "end"]:
                    label = ftr
                    labels.add(label)
    bio_labels = ["O"]
    for l in sorted(labels):
        bio_labels.append(f"B-{l}")
        bio_labels.append(f"I-{l}")
    return bio_labels

def span_to_sghead_lbls(tokens, spans):
    '''
    Helper function for df_to_sghead_ds(df)
    For an article in the polianna dataframe, takes the list of token objects and the list of span objects
    and returns list of token labels (converted from BIO to integer)
    
    :param tokens: list of token objects for a single article
    :param spans: list of span objects for a single article
    :param label2id: maps bio labels to integer labels
    '''
    token_labels = ["O"] * len(tokens)
    for spn in spans:
        if spn.layer in ["Instrumenttypes", "Policydesigncharacteristics"]:
                ftr = spn.feature
                if ftr not in ["Compliance","Reversibility", "Reference", "InstrumentType_2", "end"]:
                    start_char = spn.start
                    end_char = spn.stop
                    #
                    inside_tokens = []
                    for i, tok in enumerate(tokens):
                        tok_start = tok.start
                        tok_end = tok.stop
                        overlap = not (tok_end <= start_char or tok_start >= end_char)
                        if overlap:
                            inside_tokens.append(i)
                    if inside_tokens:
                        token_labels[inside_tokens[0]] = f"B-{ftr}"
                        for i in inside_tokens[1:]:
                            token_labels[i] = f"I-{ftr}"
    return token_labels

def df_to_sghead_ds(df):
    '''
    Converts pandas dataframe to huggingface dataset
    ***Currently ignores any articles longer than 512 tokens
    
    :param df: POLIANNA dataframe
    '''
    datapoints = []
    for artid in df.index:
        tokens = df.loc[artid,"Tokens"]
        if len(tokens) <= 512: # we'll change this eventually
            text = df.loc[artid,"Text"]
            spans = df.loc[artid,"Curation"]
            token_texts = [t.text for t in tokens]
            token_level_labels = span_to_sghead_lbls(tokens, spans)
            datapoints.append({
                "id": artid,
                "text": text,
                "tokens": token_texts,
                "ner_tags": token_level_labels
            })
    # return pd.DataFrame(datapoints)
    return Dataset.from_list(datapoints)

def create_sghead_ds(pol_dir, dir_addr):
    '''
    Creates dataset of polianna pkl, converts token and span lists to list of BIO labels converted to integers, 
    and saves new dataset and the list of labels in integer order in provided address.
    
    :param pol_dir: address of polianna pkl
    :param dir_addr: directory where to save dataset
    '''
    pol_df = df_loading(pol_dir)
    ds = df_to_sghead_ds(pol_df)
    ds.save_to_disk(dir_addr)
    print(f"Created dataset in {dir_addr}")

################################ MHEAD ################################

def span_to_mhead_lbls(feature_name, tokens, spans):
    '''
    Helper function to df_to_mhead_dataset(df)
    For an article in the dataframe, for a specific feature type in the annotated spans,
    creates a list of token labels in bio format then converts to integers.
    
    :param feature_name: Name of feature whose BIO list is being created for the article
    :param tokens: list of token objects
    :param spans: list of span objects
    :param label2id: dictionary mapping labels to integers
    '''
    token_labels = ["O"] * len(tokens)
    for spn in spans:
        if spn.feature == feature_name:
            start_char = spn.start
            end_char = spn.stop
            inside_tokens = []
            for i, tok in enumerate(tokens):
                tok_start = tok.start
                tok_end = tok.stop
                overlap = not (tok_end <= start_char or tok_start >= end_char)
                if overlap:
                    inside_tokens.append(i)
            if inside_tokens:
                token_labels[inside_tokens[0]] = f"B"
                for i in inside_tokens[1:]:
                    token_labels[i] = f"I"
    return token_labels

def df_to_mhead_ds(df):
    '''
    Converts pandas dataframe to huggingface dataset
    ***Currently ignores any articles longer than 512 tokens
    
    :param df: POLIANNA dataframe
    '''
    datapoints = []
    for artid in df.index:
        tokens = df.loc[artid,"Tokens"]
        if len(tokens) <= 512: # we'll change this eventually
            text = df.loc[artid,"Text"]
            spans = df.loc[artid,"Curation"]
            token_texts = [t.text for t in tokens]
            datapoint = {}
            datapoint['id'] = artid
            datapoint["text"] = text
            datapoint["tokens"] = token_texts
            for ftr in ["Actor", "InstrumentType", "Objective", "Resource", "Time"]:
                token_level_labels = span_to_mhead_lbls(ftr, tokens, spans)
                datapoint[f"labels_{ftr}"] = token_level_labels
            datapoints.append(datapoint)
    # return pd.DataFrame(datapoints)
    return Dataset.from_list(datapoints)

def create_mhead_ds(pol_dir, dir_addr):
    '''
    Creates dataset of polianna pkl, converts token and span lists to list of BIO labels converted to integers, 
    and saves new dataset and the list of labels in integer order in provided address.
    
    :param pol_dir: address of polianna pkl
    :param dir_addr: directory where to save dataset
    '''
    pol_df = df_loading(pol_dir)
    ds = df_to_mhead_ds(pol_df)
    ds.save_to_disk(dir_addr)
    print(f"Created dataset in {dir_addr}")

################################ Splitting ################################

def create_dsdcts(dataset, dsdct_dir, r_list=[0]):
    for r in r_list:
        td_test = dataset.train_test_split(test_size=0.2, seed=r)
        train_dev = td_test['train'].train_test_split(test_size=0.25, seed=r)
        ds_dct = DatasetDict({"train":train_dev['train'], "dev":train_dev['test'], "test":td_test['test']})
        ds_dct.save_to_disk(f"{dsdct_dir}/dsdct_r{r}")
    print(f"Created {len(r_list)} dataset(s) in {dsdct_dir}")

########
# main #
########

def main():
    cwd = os.getcwd()
    pol_dir = cwd+"/../../src/d01_data"
    ### creates whole sghead and mhead datasets from original POLIANNA database
    sghead_ds_addr = cwd+"/../inputs/sghead_ds"
    #create_sghead_ds(pol_dir, sghead_ds_addr)
    mhead_ds_addr = cwd+"/../inputs/mhead_ds"
    #create_mhead_ds(pol_dir, mhead_ds_addr)

    ### creates the dataset dictionaries for each r split from the sghead and mhead datasets
    #sghead_ds = load_from_disk(sghead_ds_addr)
    #create_dsdcts(sghead_ds, sghead_ds_addr+"dcts", list(range(3)))
    mhead_ds = load_from_disk(mhead_ds_addr)
    create_dsdcts(mhead_ds, mhead_ds_addr+"dcts", list(range(3,4)))

if __name__=="__main__":
    main()