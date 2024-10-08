import os
import pandas as pd
from PIL import Image

import random
import numpy as np
import torch
from setup import *
from utils import *
from collections import defaultdict

from tqdm import tqdm
tqdm.pandas()
import types

from overwritten_methods import hooked_merge_input_ids_with_image_features


class HiddenStateSaver():
    def __init__(self):
        self.hidden_states = defaultdict(list)

    def save_hidden_state(self, layer, hs):
        self.hidden_states[layer] = hs[layer].squeeze().detach().to(torch.float32).cpu().numpy()

    def collect_fully_connected(self, token_pos, layers):
        return np.stack([self.hidden_states[i][token_pos] for i in layers])


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
            hss.save_hidden_state(layer, mp.model.first_hidden_states)
        hidden_states.append(hss.collect_fully_connected(last_token_pos, layers))
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

def generate_qa_from_image(mp):
    df = pd.read_csv(os.path.abspath("data/full_dataset_subjects.csv"), index_col=0)
    questions = pd.read_csv(os.path.abspath("data/all_questions.csv"), index_col=0)
    df = df.merge(questions, on=["s_uri", "subject"])
    
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
            pd.DataFrame.from_records(generation_records).to_csv(f"data/qa_generations/part_{part}.csv")
            generation_records = []

    part = "last"
    pd.DataFrame.from_records(generation_records).to_csv(f"data/qa_generations/part_{part}.csv")

def generate_qa_from_text(mp):
    df = pd.read_csv(os.path.abspath("data/full_dataset_subjects.csv"), index_col=0)
    questions = pd.read_csv(os.path.abspath("data/all_questions.csv"), index_col=0)
    df = df.merge(questions, on=["s_uri", "subject"])
    
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
            pd.DataFrame.from_records(generation_records).to_csv(f"data/qa_generations_text/part_{part}.csv")
            generation_records = []

    part = "last"
    pd.DataFrame.from_records(generation_records).to_csv(f"data/qa_generations_text/part_{part}.csv")

if __name__ == "__main__":
    # mp = setup("llava_7B")
    # mp.model.vision_tower.config.output_hidden_states = True
    # mp.model._merge_input_ids_with_image_features = types.MethodType(
    #     hooked_merge_input_ids_with_image_features, mp.model
    # )
    # generate_bios_from_image(mp)

    mp = setup("llava_7B")
    generate_qa_from_text(mp)

