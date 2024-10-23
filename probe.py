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
        df[c] = df["hidden_states"].apply(lambda x: x[i] if len(layers) > 1 else x)
        tmp = torch.tensor(df[c])
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(tmp)
        tmp = scaler.transform(tmp)
        df[c] = tmp.tolist()
    df["hidden_states"] = df.apply(lambda row: [row[f"layer_{l}"] for l in layers], axis=1)
    df["hidden_states"] = df["hidden_states"].apply(lambda x: np.mean(np.array(x), axis=0).tolist())    
    return df

def eval_probe(input_type, val, test):
    probe_path = {
        "image": "probes/llava_7B_image_qa_lr_5e-05_hidden_4096_epoch_200_hs_model.pkl",
        "image_avg": "probes/llava_7B_image_avg_qa_lr_5e-05_hidden_4096_epoch_200_hs_model.pkl",
        "image_tok_subject": "probes/llava_7B_image_tok_subject_qa_lr_0.0001_hidden_4096_epoch_200_hs_model.pkl",
        "text": "probes/llava_7B_text_qa_lr_5e-05_hidden_4096_epoch_200_hs_model.pkl",
        "image_face_only": "probes/llava_7B_image_face_only_qa_lr_0.0001_hidden_4096_epoch_200_hs_model.pkl",
        "image_embeddings": "probes/llava_7B_image_embeddings_qa_lr_5e-05_hidden_4096_epoch_200_hs_model.pkl",
        "image_weighted_avg": "probes/llava_7B_image_weighted_avg_qa_lr_5e-05_hidden_4096_epoch_200_hs_model.pkl",
        "image_last_prompt_tok": "probes/llava_7B_image_last_prompt_tok_qa_lr_5e-05_hidden_4096_epoch_200_hs_model.pkl",
    }
    loaded_probe = pickle.load(open(os.path.abspath(probe_path[input_type]), 'rb')).cuda()
    full_validation_set = pd.concat([val, test])
    full_validation_set = full_validation_set.reset_index().drop("index", axis=1)

    X_full_val = torch.tensor(full_validation_set["hidden_states"].tolist(), dtype=torch.float32).cuda()
    y_full_val = torch.tensor(full_validation_set["accuracy"].tolist(), dtype=torch.float32).unsqueeze(dim=1).cuda()

    result_df, test_loss, test_pearson_corr, test_pearson_p_value = loaded_probe.validate(X_full_val, y_full_val)
    print(f"hs {generator_model_name} Pearson correlation: {test_pearson_corr:.2f} test_loss: {test_loss:.3f} test_pearson_p_value: {test_pearson_p_value}")
    result_df["subject"] = full_validation_set["subject"]
    result_df["s_uri"] = full_validation_set["s_uri"]
    result_df.to_csv(f"scores/{input_type}")


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
    
    generator_model_name = "llava_7B"
    mp = setup(generator_model_name)
    layers = list(range(int(mp.num_layers*.75)-3, int(mp.num_layers*.75)))
    
    if args.input_type == "image" and args.objective == "qa":
        dataset = qa_image_dataset()
    elif args.input_type == "image_avg" and args.objective == "qa":
        dataset = qa_image_avg_dataset()
    elif args.input_type == "image_tok_subject" and args.objective == "qa":
        dataset = qa_image_tok_subject_dataset()
    elif args.input_type == "text" and args.objective == "qa":
        dataset = qa_text_dataset()
    elif args.input_type == "image_face_only" and args.objective == "qa":
        dataset = qa_image_face_dataset()
    elif args.input_type == "image_embeddings" and args.objective == "qa":
        layers = [0]
        dataset = qa_image_embeddings_dataset()
    elif args.input_type == "image_weighted_avg" and args.objective == "qa":
        dataset = qa_image_weighted_avg_dataset()
    elif args.input_type == "image_last_prompt_tok" and args.objective == "qa":
        dataset = qa_image_last_prompt_tok_dataset()
    elif args.input_type == "image" and args.objective == "oeg":
        dataset = oeg_image_dataset()
    elif args.input_type == "text" and args.objective == "oeg":
        dataset = oeg_text_dataset()
    else:
        raise Exception("input_type, objective configuration not known")

    dataset = logits_min_max_layer_token(mp, layers, dataset, args.vocab_proj)
    train, val, test = split_into_train_eval_test(dataset)

    eval_probe(args.input_type, val, test)

    # dataset = HiddenStatesDataset(train["hidden_states"], train["accuracy"])
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # hidden_layer_size = args.hidden_layer_size
    # classifier_model_params = {
    #     "input_size": hidden_layer_size,
    #     "optimizer": "adam",
    #     "learning_rate": args.learning_rate,
    #     "max_iter": args.max_iter,
    # }
    # project = f"{args.input_type}_{args.objective}"
    # label = "vocab" if args.vocab_proj else "hs"
    # run_name = f"lr_{classifier_model_params['learning_rate']}_hidden_{classifier_model_params['input_size']}_epoch_{classifier_model_params['max_iter']}_{label}" 
    # wandb.init(project=project, name=run_name, config=classifier_model_params)

    # # Build the MLPRegressor model and train it
    # model = MLPRegressor(**classifier_model_params).cuda()
    # model.fit(dataloader, train["accuracy"], val["hidden_states"], val["accuracy"]) 
    # with open(f"probes/{generator_model_name}_{project}_{run_name}_model.pkl",'wb') as f:
    #     model.set_to_best_weights()
    #     pickle.dump(model, f)


