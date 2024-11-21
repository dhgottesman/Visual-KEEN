import argparse

import os
import pandas as pd
from PIL import Image

import numpy as np
import torch
from setup import *
from utils import *
from collections import defaultdict

from tqdm import tqdm
tqdm.pandas()
import types

import matplotlib.pyplot as plt

from overwritten_methods import hooked_merge_input_ids_with_image_features


class HiddenStateSaver():
    def __init__(self):
        self.hidden_states = defaultdict(list)

    def save_hidden_state(self, layer, hs):
        self.hidden_states[layer] = hs[layer].squeeze().detach().to(torch.float32).cpu().numpy()
    
    def collect_hidden_states(self, token_pos, layers):
        return np.stack([self.hidden_states[layer][token_pos] for layer in layers])

def extract_func_tok_subject(mp, inputs, layers, image_token_range=None):
    hss = HiddenStateSaver()
    s_range = find_token_range(mp.processor.tokenizer, inputs["input_ids"][0], "subject")
    last_token_pos = s_range[0]
    for layer in layers:
        hss.save_hidden_state(layer, mp.model.first_hidden_states)
    return hss.collect_hidden_states(last_token_pos, layers)

def extract_func_tok_image(mp, inputs, layers, image_token_range=None):
    hss = HiddenStateSaver()
    s_range = find_token_range(mp.processor.tokenizer, inputs["input_ids"][0], "image\n")
    last_token_pos = s_range[0]
    for layer in layers:
        hss.save_hidden_state(layer, mp.model.first_hidden_states)
    return hss.collect_hidden_states(last_token_pos, layers)

def extract_func_average_image(mp, inputs, layers, image_token_range=None):
    hss = HiddenStateSaver()
    if image_token_range is None:
        image_token_range = torch.where(mp.model.image_to_overwrite[0] == True)[0]
    for layer in layers:
        hss.save_hidden_state(layer, mp.model.first_hidden_states)
    hidden_states = []
    for pos in image_token_range:
        hidden_states.append(hss.collect_hidden_states(pos, layers))
    return np.mean(np.stack(hidden_states), axis=0)

def extract_func_weighted_average_image(mp, inputs, layers, image_token_range=None):
    hss = HiddenStateSaver()
    image_token_range = torch.where(mp.model.image_to_overwrite[0] == True)[0]
    for layer in layers:
        hss.save_hidden_state(layer, mp.model.first_hidden_states)
    hidden_states = []
    for pos in image_token_range:
        hidden_states.append(hss.collect_hidden_states(pos, layers))

    stacked_hidden_states = np.stack(hidden_states)
    num_tokens = stacked_hidden_states.shape[0]
    
    weights = np.arange(1, num_tokens + 1)
    weights = weights / weights.sum()
    weighted_avg = np.average(stacked_hidden_states, axis=0, weights=weights)
    return weighted_avg

def extract_image_embeddings(mp, inputs, layers, image_token_range=None):
    return extract_func_average_image(mp, inputs, [0])

def extract_last_prompt_token(mp, inputs, layers, image_token_range=None):
    hss = HiddenStateSaver()
    last_token_pos = -1
    for layer in layers:
        hss.save_hidden_state(layer, mp.model.first_hidden_states)
    return hss.collect_hidden_states(last_token_pos, layers)  

def extract_face_image_inputs(extract_func, output_dir, start, end):
    layers = list(range(int(mp.num_layers*.75)-3, int(mp.num_layers*.75)))
    df = pd.read_csv(os.path.abspath("data/full_dataset_subjects.csv"), index_col=0)
    hidden_states = []
    for i, row in tqdm(df.iterrows(), desc="row"):
        if i < start:
            continue
        if i == end:
            # Assumes already wrote out last batch, i.e. if end == 200, that batch i=[0,199] would have already been written.
            return
        image_file = os.path.join("/home/morg/dataset/imdb_wiki/wiki", row["path"])
        raw_image = Image.open(image_file)
        inputs = mp.processor(images=raw_image, text=GENERATE_BIOS_PROMPTS, return_tensors='pt').to(0, torch.float16)

        _ = mp.model.generate(**inputs, max_new_tokens=1, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)
        image_token_range = []
        for x, y, w, h in mp.processor.image_processor.faces:
            image_token_range += get_embedding_indices_from_bounding_box(x, y, x + w, y + h, mp.patch_size, mp.num_patches)
        if len(image_token_range) == 0:
            hidden_states.append(torch.full((3, 4096), -1))
            print(f"No faces found for image {i} {image_file}.")
        else:
            hidden_states.append(extract_func(mp, inputs, layers, image_token_range))
        del mp.model.first_hidden_states
        del mp.processor.image_processor.faces

        if len(hidden_states) == 200:
            part = i // 200
            torch.save(torch.tensor(np.stack([hidden_states])).squeeze(), f"data/{output_dir}/part_{part}.pt")
            hidden_states = []

    part = "last"
    torch.save(torch.tensor(np.stack([hidden_states])).squeeze(), f"data/{output_dir}/part_{part}.pt")

def extract_additional_image_inputs(extract_func, output_dir, start, end):
    layers = list(range(int(mp.num_layers*.75)-3, int(mp.num_layers*.75)))
    df = pd.read_csv(os.path.abspath("data/full_dataset_subjects.csv"), index_col=0)
    hidden_states = []
    for i, row in tqdm(df.iterrows(), desc="row"):
        if i < start:
            continue
        if i == end:
            # Assumes already wrote out last batch, i.e. if end == 200, that batch i=[0,199] would have already been written.
            return
        image_file = os.path.join("/home/morg/dataset/imdb_wiki/wiki", row["path"])
        raw_image = Image.open(image_file)
        inputs = mp.processor(images=raw_image, text=GENERATE_BIOS_PROMPTS, return_tensors='pt').to(0, torch.float16)

        _ = mp.model.generate(**inputs, max_new_tokens=1, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)
        hidden_states.append(extract_func(mp, inputs, layers))
        del mp.model.first_hidden_states

        if len(hidden_states) == 200:
            part = i // 200
            torch.save(torch.tensor(np.stack([hidden_states])).squeeze(), f"data/{output_dir}/part_{part}.pt")
            hidden_states = []

    part = "last"
    torch.save(torch.tensor(np.stack([hidden_states])).squeeze(), f"data/{output_dir}/part_{part}.pt")

def generate_bios_from_image(mp):
    layers = list(range(int(mp.num_layers*.75)-3, int(mp.num_layers*.75)))
    df = pd.read_csv(os.path.abspath("data/full_dataset_subjects.csv"), index_col=0)
    generation_records = []
    hidden_states = []
    for i, row in tqdm(df.iterrows(), desc="row"):
        image_file = os.path.join("/home/morg/dataset/imdb_wiki/wiki", row["path"])
        raw_image = Image.open(image_file)
        inputs = mp.processor(images=raw_image, text=GENERATE_BIOS_PROMPTS, return_tensors='pt').to(0, torch.float16)

        output = mp.model.generate(**inputs, max_new_tokens=500, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)

        generated_text = mp.processor.batch_decode(output.sequences, skip_special_tokens=True)
        generation_records.append({
          "s_uri": row["s_uri"],
          "subject": row["subject"],
          "generation": generated_text[0].split("ASSISTANT:")[-1]
        })

        image_token_range = torch.where(mp.model.image_to_overwrite[0] == True)[0]
        last_token_pos = image_token_range[-1].item()

        hss = HiddenStateSaver()
        for layer in layers:
            hss.save_image_hidden_state(layer, mp.model.first_hidden_states)
        hidden_states.append(hss.collect_hidden_states(last_token_pos, layers))
        del mp.model.first_hidden_states

        if len(hidden_states) == 200:
            part = i // 200
            torch.save(torch.tensor(np.stack([hidden_states])).squeeze(), f"data/input_hidden_states/part_{part}.pt")
            hidden_states = []
        
            pd.DataFrame.from_records(generation_records).to_csv(f"data/oeg_generations/part_{part}.csv")
            generation_records = []

    part = "last"
    torch.save(torch.tensor(np.stack([hidden_states])).squeeze(), f"data/input_hidden_states/part_{part}.pt")
    pd.DataFrame.from_records(generation_records).to_csv(f"data/oeg_generations/part_{part}.csv")

def generate_bios_from_text(mp):
    layers = list(range(int(mp.num_layers*.75)-3, int(mp.num_layers*.75)))
    df = pd.read_csv(os.path.abspath("data/full_dataset_subjects.csv"), index_col=0)
    generation_records = []
    hidden_states = []
    for i, row in tqdm(df.iterrows(), desc="row"):
        if i < 2000:
            continue
        subject = row["subject"]
        prompt = f"USER: \nGenerate a biography about {subject}\nASSISTANT:"
        inputs = mp.processor.tokenizer(text=prompt, return_tensors='pt')
        inputs["input_ids"] = inputs["input_ids"].cuda()
        inputs["attention_mask"] = inputs["attention_mask"].cuda()

        output = mp.model.generate(**inputs, max_new_tokens=500, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)
        try:
            s_range = find_token_range(mp.processor.tokenizer, inputs["input_ids"][0], subject.replace(" ", ""))
            last_token_pos = s_range[-1] - 1
        except ValueError:
            s_range = find_token_range(mp.processor.tokenizer, inputs["input_ids"][0], "\nASSISTANT:")
            last_token_pos = s_range[0] - 1

        generated_text = mp.processor.batch_decode(output.sequences, skip_special_tokens=True)
        generation_records.append({
          "s_uri": row["s_uri"],
          "subject": row["subject"],
          "generation": generated_text[0].split("ASSISTANT:")[-1]
        })

        hss = HiddenStateSaver()
        for layer in layers:
            hss.save_hidden_state(layer, mp.model.first_hidden_states)
        hidden_states.append(hss.collect_hidden_states(last_token_pos, layers))
        del mp.model.first_hidden_states

        if len(hidden_states) == 200:
            part = i // 200
            torch.save(torch.tensor(np.stack([hidden_states])).squeeze(), f"data/input_hidden_states_text/part_{part}.pt")
            hidden_states = []
        
            pd.DataFrame.from_records(generation_records).to_csv(f"data/oeg_generations_text/part_{part}.csv")
            generation_records = []

    part = "last"
    torch.save(torch.tensor(np.stack([hidden_states])).squeeze(), f"data/input_hidden_states_text/part_{part}.pt")
    pd.DataFrame.from_records(generation_records).to_csv(f"data/oeg_generations_text/part_{part}.csv")

def generate_qa_from_image(mp):
    df = pd.read_csv(os.path.abspath("data/full_dataset_subjects.csv"), index_col=0)
    # questions = pd.read_csv(os.path.abspath("data/all_questions.csv"), index_col=0)
    questions = pd.read_csv(os.path.abspath("data/more_questions.csv"), index_col=0)
    df = df.merge(questions, on=["s_uri", "subject"])
    df = df.drop_duplicates("question")
    df = df.reset_index().drop("index", axis=1)
    
    generation_records = []
    for i, row in tqdm(df.iterrows(), desc="row"):
        image_file = os.path.join("/home/morg/dataset/imdb_wiki/wiki", row["path"])
        raw_image = Image.open(image_file)
        question = row["question_for_image"]
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = mp.processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

        output = mp.model.generate(**inputs, max_new_tokens=100, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)

        generated_text = mp.processor.batch_decode(output.sequences, skip_special_tokens=True)
        generation_records.append({
          "s_uri": row["s_uri"],
          "subject": row["subject"],
          "question_for_image": question,
          "generation": generated_text[0].split("ASSISTANT:")[-1]
        })

        if len(generation_records) == 5000:
            part = i // 5000        
            pd.DataFrame.from_records(generation_records).to_csv(f"data/qa_generations/part_more_{part}.csv")
            generation_records = []

    part = "last"
    pd.DataFrame.from_records(generation_records).to_csv(f"data/qa_generations/part_more_{part}.csv")

def generate_qa_from_text(mp):
    df = pd.read_csv(os.path.abspath("data/full_dataset_subjects.csv"), index_col=0)
    questions = pd.read_csv(os.path.abspath("data/all_questions.csv"), index_col=0)
    df = df.merge(questions, on=["s_uri", "subject"])
    df = df.drop_duplicates("question")
    df = df.reset_index().drop("index", axis=1)

    generation_records = []
    for i, row in tqdm(df.iterrows(), desc="row"):
        question = row["question"]
        prompt = f"USER: \n{question}\nASSISTANT:"
        inputs = mp.processor.tokenizer(text=prompt, return_tensors='pt')
        inputs["input_ids"] = inputs["input_ids"].cuda()
        inputs["attention_mask"] = inputs["attention_mask"].cuda()

        output = mp.model.generate(**inputs, max_new_tokens=100, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)

        generated_text = mp.processor.batch_decode(output.sequences, skip_special_tokens=True)
        generation_records.append({
          "s_uri": row["s_uri"],
          "subject": row["subject"],
          "question": question,
          "generation": generated_text[0].split("ASSISTANT:")[-1]
        })

        if len(generation_records) == 5000:
            part = i // 5000        
            pd.DataFrame.from_records(generation_records).to_csv(f"data/qa_generations_text/part_more_{part}.csv")
            generation_records = []

    part = "last"
    pd.DataFrame.from_records(generation_records).to_csv(f"data/qa_generations_text/part_more_{part}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llava_7B")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    args = parser.parse_args()  

    mp = setup(args.model_name)
    mp.model.vision_tower.config.output_hidden_states = True
    mp.model._merge_input_ids_with_image_features = types.MethodType(
        hooked_merge_input_ids_with_image_features, mp.model
    )

    extract_additional_image_inputs(extract_last_prompt_token, "input_hidden_states_last_prompt_token", args.start, args.end)

    # Additional experiments
    #
    # extract_additional_image_inputs(extract_func_tok_subject, "input_hidden_states_image_tok_subject", args.start, args.end)
    # extract_face_image_inputs(extract_func_average_image, "input_hidden_states_only_face_image_avg", args.start, args.end)
    # extract_additional_image_inputs(extract_func_tok_image, "input_hidden_states_image_tok_image", args.start, args.end)
    
    # mp = setup(args.model_name)
    # mp.model.vision_tower.config.output_hidden_states = True
    # mp.model._merge_input_ids_with_image_features = types.MethodType(
    #     hooked_merge_input_ids_with_image_features, mp.model
    # )
    # generate_bios_from_image(mp)
    # mp = setup("llava_7B")
    # generate_bios_from_text(mp)


