import argparse

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['CURL_CA_BUNDLE'] = ''

from sklearn import preprocessing
import wandb
from tqdm import tqdm
from mlp_regressor import MLPRegressor
import torch
import pickle

from torch.utils.data import DataLoader

from setup import setup, MODEL_NAME_PATH
from datasets import Dataset

from tqdm import tqdm
tqdm.pandas()

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import *

# Deterministic Run
import random
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


class HiddenStatesDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.df = X_train
        self.labels = y_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        hidden_states = self.df.iloc[idx]
        accuracy = self.labels.iloc[idx]
        return torch.tensor(hidden_states, dtype=torch.float32), torch.tensor(accuracy, dtype=torch.float32)

def logits_min_max_layer_token(mp, layers, df, vocab_proj):
    def _extract_features(hidden_states, layers):
        new_hs = []
        for i, layer in enumerate(layers):
            if vocab_proj:
                hs = torch.tensor(hidden_states[i])
                hs = mp.vocabulary_projection_function(hs, layer)
            new_hs.append(hs.detach().cpu().numpy().tolist())
        return new_hs
    if vocab_proj:
        df["hidden_states"] = df["hidden_states"].apply(_extract_features)
    for i, l in enumerate(layers):
        c = f"layer_{l}"
        df[c] = df["hidden_states"].apply(lambda x: x[i])
        tmp = torch.tensor(df[c])
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(tmp)
        tmp = scaler.transform(tmp)
        df[c] = tmp.tolist()
    df["hidden_states"] = df.apply(lambda row: [row[f"layer_{l}"] for l in layers], axis=1)
    df["hidden_states"] = df["hidden_states"].apply(lambda x: np.mean(np.array(x), axis=0).tolist())    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--vocab_proj', action="store_true")
    parser.add_argument('--hidden_layer_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--max_iter', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--layers', type=str)
    parser.add_argument('--input_type', type=str)
    parser.add_argument('--objective', type=str)
    args = parser.parse_args()  

    if args.input_type == "image" and args.objective == "qa":
        dataset = qa_image_dataset()
    elif args.input_type == "image_avg" and args.objective == "qa":
        dataset = qa_image_avg_dataset()
    elif args.input_type == "image_tok_subject" and args.objective == "qa":
        dataset = qa_image_tok_subject_dataset()
    elif args.input_type == "text" and args.objective == "qa":
        dataset = qa_text_dataset()
    elif args.input_type == "image" and args.objective == "oeg":
        dataset = oeg_image_dataset()
    elif args.input_type == "text" and args.objective == "oeg":
        dataset = oeg_text_dataset()
    else:
        raise Exception("input_type, objective configuration not known")

    generator_model_name = "llava_7B"
    mp = setup(generator_model_name)
    layers = list(range(int(mp.num_layers*.75)-3, int(mp.num_layers*.75)))
    dataset = logits_min_max_layer_token(mp, layers, dataset, args.vocab_proj)
    train, val, test = split_into_train_eval_test(dataset)

    dataset = HiddenStatesDataset(train["hidden_states"], train["accuracy"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    hidden_layer_size = args.hidden_layer_size
    classifier_model_params = {
        "input_size": hidden_layer_size,
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "max_iter": args.max_iter,
    }
    project = f"{args.input_type}_{args.objective}"
    label = "vocab" if args.vocab_proj else "hs"
    run_name = f"lr_{classifier_model_params['learning_rate']}_hidden_{classifier_model_params['input_size']}_epoch_{classifier_model_params['max_iter']}_{label}" 
    wandb.init(project=project, name=run_name, config=classifier_model_params)

    # Build the MLPRegressor model and train it
    model = MLPRegressor(**classifier_model_params).cuda()
    model.fit(dataloader, train["accuracy"], val["hidden_states"], val["accuracy"]) 
    with open(f"probes/{generator_model_name}_{project}_{run_name}_model.pkl",'wb') as f:
        model.set_to_best_weights()
        pickle.dump(model, f)


