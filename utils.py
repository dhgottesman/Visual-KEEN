import os
import torch
from ast import literal_eval
import numpy as np
import pandas as pd


GENERATE_BIOS_PROMPTS = ["USER: <image>\nIdentify by name and generate a biography about the subject of this image\nASSISTANT:"]


def split_dataset_into_train_val_test(dataset, features="hidden_states"):
    return None

def document_prefix(subject):
    return f"This document describes {subject}"

def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]

def find_token_range(tokenizer, token_array, substring):
    """Find the tokens corresponding to the given substring in token_array."""
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)

def sample_subjects():
    df = pd.read_csv(os.path.abspath("data/meta.csv"), index_col=0)
    df = df.dropna()
    df = df.sort_values(by="s_pop", ascending=False)

    head = df.head(5000)
    tail = df.tail(5000)
    torso = df.sample(1000)

    df = pd.concat([head, torso, tail]).drop_duplicates("s_uri")
    df = df.reset_index().drop("index", axis=1)
    df.to_csv("data/full_dataset_subjects.csv")

def _load_csv_df_from_dir(directory, files):
  files = [f for f in files if f[-4:] == ".csv"]
  df = None
  for f in files:
      fp = os.path.join(directory, f)
      if df is None:
          df = pd.read_csv(fp, index_col=0)
      else:
          df = pd.concat([df, pd.read_csv(fp, index_col=0)])
  return df.reset_index()

def _load_tensor_from_files(directory, files):
    tensors = []
    for filename in files:
        if filename.endswith('.pt'):
            tensor = torch.load(os.path.join(directory, filename))
            tensors.append(tensor) 
    return torch.cat(tensors, dim=0)

def _sort_key(part):
    if part.split('.')[0] == 'part_last':
        return float('inf')
    else:
        return int(part.split('_')[1].split('.')[0])

def load_oeg_image_generations():
    directory = os.path.abspath("data/oeg_generations")
    files = sorted(os.listdir(directory), key=_sort_key)
    return _load_csv_df_from_dir(directory, files)

def load_qa_image_generations():
    directory = os.path.abspath("data/qa_generations")
    files = sorted(os.listdir(directory), key=_sort_key)
    return _load_csv_df_from_dir(directory, files)

def load_inputs():
    directory = os.path.abspath("data/input_hidden_states")
    files = sorted(os.listdir(directory), key=_sort_key)
    return _load_tensor_from_files(directory, files)

def qa_accuracy():
    df = pd.read_csv("PATH TO GENERATED PASSAGES", index_col=0)
    
    questions = pd.read_csv(os.path.abspath("data/all_questions.csv"), index_col=0)
    questions["possible_answers"] = questions["possible_answers"].apply(lambda x: literal_eval(x))
    questions = questions.rename(columns={"subj": "subject"})

    df = df.merge(questions, on="question")

    def label_generation(generation, answers):
        for answer in answers:
            if answer.lower() in generation.lower():
                return 3
        for hedged_answer in ["nobody knows", "I'm sorry", "I can't seem to find the answer", "you help me", "anyone help me", "I'm not sure", "I don't know", "I am not sure", "I\'m not sure", "I'm not entirely sure", "Could you please provide more", "could provide more information", "provide more context", "clarify your question"]:
            if hedged_answer.lower() in generation.lower():
                return 2
        if hedged_answer == "":
            return 2
        return 1
    
    def binary_label(label, class_label):
        return 1 if label == class_label else 0

    df["generation_label"] = df.apply(lambda row: label_generation(row["generation"], row["possible_answers"]), axis=1)
    # Multiple answers for each question, if one of them is correct then mark the question as correct.
    idx = df.groupby(['subject', 's_uri', 'prop'])["generation_label"].idxmax()
    df = df.iloc[idx]

    # Compute correct, hedged, mistake accuracy.
    df["correct"] = df["generation_label"].apply(lambda x: binary_label(x, 3))
    df["hedge"] = df["generation_label"].apply(lambda x: binary_label(x, 2))
    df["mistake"] = df["generation_label"].apply(lambda x: binary_label(x, 1))

    result_df = df.groupby(['subject', 's_uri', "label"]).agg(
        total_examples=('generation_label', 'count'),
        accuracy=('correct', 'mean'),
        hedged_frac=('hedge', 'mean'),
        mistake_frac=('mistake', 'mean')
    ).reset_index()

    result_df = result_df[result_df["total_examples"] > 1]
    return result_df[["subject", "accuracy", "total_examples", "hedged_frac", "mistake_frac"]]
