import os
import torch
from ast import literal_eval
import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

GENERATE_BIOS_PROMPTS = ["USER: <image>\nIdentify by name and generate a biography about the subject of this image\nASSISTANT:"]


def anti_join(df1, df2, columns):
  df = df1.merge(df2[columns], how='left', on=columns, indicator='source')
  return df[df["source"] == 'left_only'].drop('source', axis=1)

def split_into_train_eval_test(df):
    seed = 42
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    split_index_train = int(0.65 * len(df))
    train = df.iloc[:split_index_train]
    split_index_eval = split_index_train + int(0.15 * len(df))
    val = df.iloc[split_index_train:split_index_eval]
    test = df.iloc[split_index_eval:]
    return train, val, test

def qa_image_dataset():
    inputs = load_image_inputs()
    subjects = pd.read_csv(os.path.abspath("data/full_dataset_subjects.csv"), index_col=0)
    subjects["hidden_states"] = inputs.tolist()

    accuracy = qa_accuracy_image()
    return subjects.merge(accuracy, on=["subject", "s_uri"])

def qa_text_dataset():
    inputs = load_text_inputs()
    subjects = pd.read_csv(os.path.abspath("data/full_dataset_subjects.csv"), index_col=0)
    subjects["hidden_states"] = inputs.tolist()

    accuracy = qa_accuracy_text()
    return subjects.merge(accuracy, on=["subject", "s_uri"])

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

def load_oeg_text_generations():
    directory = os.path.abspath("data/oeg_generations_text")
    files = sorted(os.listdir(directory), key=_sort_key)
    return _load_csv_df_from_dir(directory, files)

def load_qa_text_generations():
    directory = os.path.abspath("data/qa_generations_text")
    files = sorted(os.listdir(directory), key=_sort_key)
    return _load_csv_df_from_dir(directory, files)

def load_image_inputs():
    directory = os.path.abspath("data/input_hidden_states")
    files = sorted(os.listdir(directory), key=_sort_key)
    return _load_tensor_from_files(directory, files)

def load_text_inputs():
    directory = os.path.abspath("data/input_hidden_states_text")
    files = sorted(os.listdir(directory), key=_sort_key)
    return _load_tensor_from_files(directory, files)

def _qa_accuracy(df):    
    def label_generation(generation, answers):
        for answer in answers:
            if answer.isupper() and answer in generation:
                return 3
            elif not answer.isupper() and answer.lower() in generation.lower():
                return 3
        for hedged_answer in ["nobody knows", "I'm sorry", "I can't seem to find the answer", "you help me", "anyone help me", "I'm not sure", "I don't know", "I am not sure", "I\'m not sure", "I'm not entirely sure", "Could you please provide more", "could provide more information", "provide more context", "clarify your question"]:
            if hedged_answer.lower() in generation.lower():
                return 2
        if hedged_answer == "":
            return 2
        return 1
    
    def binary_label(label, class_label):
        return 1 if label == class_label else 0

    df["generation_label"] = df.progress_apply(lambda row: label_generation(row["generation"], row["possible_answers"]), axis=1)
    # Multiple answers for each question, if one of them is correct then mark the question as correct.
    idx = df.groupby(['subject', 's_uri', 'r_uri'])["generation_label"].idxmax()
    df = df.iloc[idx]

    # Compute correct, hedged, mistake accuracy.
    df["correct"] = df["generation_label"].apply(lambda x: binary_label(x, 3))
    df["hedge"] = df["generation_label"].apply(lambda x: binary_label(x, 2))
    df["mistake"] = df["generation_label"].apply(lambda x: binary_label(x, 1))

    result_df = df.groupby(['subject', 's_uri']).agg(
        total_examples=('generation_label', 'count'),
        accuracy=('correct', 'mean'),
        hedged_frac=('hedge', 'mean'),
        mistake_frac=('mistake', 'mean')
    ).reset_index()

    result_df = result_df[result_df["total_examples"] > 1]
    return result_df[["subject", "s_uri", "accuracy", "total_examples", "hedged_frac", "mistake_frac"]]

def qa_accuracy_image():
    df = load_qa_image_generations()
    questions = pd.read_csv(os.path.abspath("data/all_questions.csv"), index_col=0)
    questions["possible_answers"] = questions["possible_answers"].progress_apply(lambda x: literal_eval(x))
    # Duplicates from multiple answers per question and multiple template variations per relation.
    df = df.merge(questions, on=["subject", "question_for_image", "s_uri"])
    df = df.reset_index().drop("index", axis=1)
    return _qa_accuracy(df)

def qa_accuracy_text():
    df = load_qa_text_generations()
    questions = pd.read_csv(os.path.abspath("data/all_questions.csv"), index_col=0)
    questions["possible_answers"] = questions["possible_answers"].progress_apply(lambda x: literal_eval(x))
    df = df.merge(questions, on=["subject", "question", "s_uri"])
    return _qa_accuracy(df)

print("here")