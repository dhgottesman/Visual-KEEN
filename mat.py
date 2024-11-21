import numpy as np
from scipy.io import loadmat
import pandas as pd
import json


cols = ['subject', 'path', 'gender', 'face_score1', 'face_score2']

wiki_mat = '/home/morg/dataset/imdb_wiki/wiki/wiki.mat'

wiki_data = loadmat(wiki_mat)

del wiki_mat

wiki = wiki_data['wiki']
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
wiki_name = wiki[0][0][4][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

wiki_path = []
for path in wiki_full_path:
    wiki_path.append(path[0])

wiki_genders = []
for n in range(len(wiki_gender)):
    if wiki_gender[n] == 1:
        wiki_genders.append('male')
    else:
        wiki_genders.append('female')

wiki_dob = []
for file in wiki_path:
    wiki_dob.append(file.split('_')[2])

final_wiki = np.vstack((wiki_name, wiki_path, wiki_genders, wiki_face_score1, wiki_face_score2)).T
final_wiki_df = pd.DataFrame(final_wiki)
final_wiki_df.columns = cols

meta = final_wiki_df
meta['face_score1'] = pd.to_numeric(meta['face_score1'], errors='coerce')
meta = meta[~np.isinf(meta['face_score1']) & pd.notna(meta['face_score1'])]
meta = meta[pd.isna(meta['face_score2'])]

meta = meta.sample(frac=1)
meta["subject"] = meta["subject"].apply(lambda x: str(x[0]) if len(x) > 0 else np.nan)
meta = meta.dropna(subset=["subject"])
meta = meta.drop(columns="face_score2")

with open("/home/morg/students/yoavgurarieh/qids.json", "r") as f:
    names_to_ids = json.load(f)

meta["s_uri"] = meta["subject"].apply(lambda x: names_to_ids.get(x, np.nan))
meta = meta.dropna()

meta = meta.sort_values(by="face_score1", ascending=False)
meta = meta.reset_index(drop="index")

meta.to_csv('meta.csv', index=False)
