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

from train_classifier_new import qa_accuracy
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

def logits_min_max_layer_token(mt, device, prompt_func, layers, df, vocab_proj):
    def _extract_features(subject):
        prompt = prompt_func(subject)
        with torch.no_grad():
            inp = mt.tokenizer(prompt, return_tensors="pt").to(device)
            output = mt.model(**inp, output_hidden_states = True) 
        hidden_states = []
        for layer in layers:
            hs = output["hidden_states"][layer][0][-1]
            if vocab_proj:
                hs = mt.vocabulary_projection_function(hs, layer)
            hidden_states.append(hs.detach().cpu().numpy().tolist())
        return hidden_states

    df["hidden_states"] = df["subject"].progress_apply(_extract_features)
    n_layers = len(df["hidden_states"].iloc[0])
    for i in range(n_layers):
        c = f"layer_{i}"
        df[c] = df["hidden_states"].apply(lambda x: x[i])
        tmp = torch.tensor(df[c])
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(tmp)
        tmp = scaler.transform(tmp)
        df[c] = tmp.tolist()
    df["hidden_states"] = df.apply(lambda row: [row[f"layer_{i}"] for i in range(n_layers)], axis=1)
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
    args = parser.parse_args()  

    generator_model_name = args.model_name
    print("Loading", generator_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    mt = setup(MODEL_NAME_PATH[generator_model_name])
    mt.model = mt.model.to(device)

    prompt_func = lambda x: document_prefix(x)
    layers = list(range(int(mt.num_layers*.75)-5, int(mt.num_layers*.75)))
    
    dataset = qa_accuracy(generator_model_name)
    dataset = dataset[["subject", "accuracy", "total_examples"]]
    dataset = dataset.reset_index().drop("index",axis=1)
    dataset = logits_min_max_layer_token(mt, device, prompt_func, layers, dataset, args.vocab_proj)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_into_train_val_test(dataset)

    dataset = HiddenStatesDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    hidden_layer_size = args.cutoff if args.cutoff > -1 else args.hidden_layer_size
    classifier_model_params = {
        "input_size": len(layers),
        "output_size": 1,
        "hidden_layer_size": hidden_layer_size,
        "hidden_activation": "relu",
        "last_activation": "sigmoid",
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "max_iter": args.max_iter,
        "device": device,
    }
    project = f"{generator_model_name}_probe"
    label = "vocab" if args.vocab_proj else "hs"
    run_name = f"{generator_model_name}_hidden_states_{args.layers}_lr_{classifier_model_params['learning_rate']}_hidden_{classifier_model_params['hidden_layer_size']}_epoch_{classifier_model_params['max_iter']}_{label}{cutoff_label}_min_max_avg_batched" 
    wandb.init(project=project, name=run_name, config=classifier_model_params)

    # Build the MLPRegressor model and train it
    model = MLPRegressor(**classifier_model_params).to(device)
    model.fit(dataloader, y_train, X_val, y_val) 
    with open(f"probes/{generator_model_name}_{project}_{run_name}_model.pkl",'wb') as f:
        model.set_to_best_weights()
        pickle.dump(model, f)


