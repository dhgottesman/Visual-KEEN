from ast import literal_eval
import pandas as pd


def split_dataset_into_train_val_test(dataset, features="hidden_states"):
    return None

def document_prefix(subject):
    return f"This document describes {subject}"

def qa_accuracy():
    df = pd.read_csv("PATH TO GENERATED PASSAGES", index_col=0)
    
    questions = pd.read_csv("PATH TO QUESTIONS FROM WIKIDATA", index_col=0)
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