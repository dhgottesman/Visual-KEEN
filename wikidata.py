import os
import pandas as pd
from ast import literal_eval

from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
from urllib.parse import urlparse

from tqdm import tqdm
tqdm.pandas()

class SPARQL:
    def __init__(self):
        self.agent = "'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'"
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=self.agent)
        self.sparql.setReturnFormat(JSON)

    def parse_value(self, value):
        parsed_uri = urlparse(value)
        if all([parsed_uri.scheme, parsed_uri.netloc]):
            return parsed_uri.path.split('/')[-1]
        return value 

    def execute(self, query):
        records = []     
        try:
            self.sparql.setQuery(query)
            responses = self.sparql.query().convert()
            for response in responses['results']['bindings']:
                record = {}
                for key in response:
                    record[key] = self.parse_value(response[key]['value'])
                records.append(record)
            if records == 0:
                print("request failed")
        except Exception as e:
            print(e)
        return pd.DataFrame(records)

def get_all_properties():
    def _query(relation_ids): 
        return f'''
SELECT ?item ?itemLabel ?wd ?wdLabel ?ps_ ?ps_Label WHERE {{
  VALUES ?item {{ 
    {" ".join([f"wd:{id}" for id in relation_ids])}
  }}
  ?item ?p ?statement .
  ?statement ?ps ?ps_ .
  ?wd wikibase:claim ?p .
  ?wd wikibase:statementProperty ?ps .
  
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}
'''
    sparql = SPARQL()
    
    df = pd.read_csv("meta.csv")
    subjects = df["s_uri"].to_list()
    subject_chunks = [subjects[i:i+20] for i in range(0, len(subjects), 20)]
    
    df = pd.concat([sparql.execute(_query(chunk)) for chunk in tqdm(subject_chunks)])
    df = df[~df["wdLabel"].str.contains(r"ID|category|template|username|instance of|gallery|article|handle|url|wiki|copyright|classification|website|described|tag|archive|reddit|profile|image|list|file", case=False, na=False)]
    df = df[~df["ps_Label"].str.contains(r'\d', na=False)]
    df = df[["item", "itemLabel", "wd", "wdLabel", "ps_", "ps_Label"]]
    df = df.rename(
        columns = {
            "item": "subject",
            "itemLabel": "s_uri",
            "wd": "r_uri",
            "wdLabel": "relation",
            "ps_": "a_uri",
            "ps_Label": "attribute",
        }
    )
    df.to_csv("data/subjects_to_relations.csv", index=False)

def get_aliases():
    def _query(uris): 
        return f'''
SELECT ?uri ?alias
WHERE {{
    {{VALUES ?uri {{ {" ".join([f"wd:{uri}" for uri in uris])} }} }}
    ?uri skos:altLabel ?alias.
    FILTER(LANG(?alias) = "en")
}}
''' 
    sparql = SPARQL()

    df = pd.read_csv(os.path.abspath("data/subjects_to_relations.csv"))
    uris = list(set(df["s_uri"].tolist() + df["a_uri"].tolist()))
    uri_chunks = [uris[i:i+100] for i in range(0, len(uris), 100)]

    aliases = pd.concat([sparql.execute(_query(chunk)) for chunk in tqdm(uri_chunks)])
    aliases = aliases.groupby("uri")["alias"].agg(list).reset_index(name="aliases")
    aliases.to_csv("data/all_aliases.csv")

def attribute_type():
    def _query(uris):
        return f'''
SELECT ?uri ?typeLabel
WHERE {{
  {{VALUES ?uri {{ {" ".join([f"wd:{uri}" for uri in uris])} }} }}
  ?uri wdt:P31 ?type.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}   
''' 
    sparql = SPARQL()
    
    df = pd.read_csv(os.path.abspath("data/subjects_to_relations.csv"))
    uris = df["a_uri"].drop_duplicates().to_list()
    uri_chunks = [uris[i:i+100] for i in range(0, len(uris), 100)]
    a_types = pd.concat([sparql.execute(_query(chunk)) for chunk in tqdm(uri_chunks)])
    a_types = a_types.groupby("uri")["typeLabel"].agg(list).reset_index(name="a_type")
    a_types['a_type'] = a_types['a_type'].apply(lambda x: x if type(x) == list else [])
    a_types.to_csv("data/complete_attribute_types.csv", index=False)

def aggregate_triples():
    df = pd.read_csv(os.path.abspath("data/subjects_to_relations.csv"))
    subjects = pd.read_csv("/home/morg/students/gottesman3/visual-KEEN/subject_to_generate_questions_for.csv", index_col=0)
    df = df.merge(subjects, on="s_uri")
    aliases = pd.read_csv("/home/morg/students/gottesman3/visual-KEEN/data/all_aliases.csv", index_col=0)
    aliases["aliases"] = aliases["aliases"].apply(lambda x: literal_eval(x))
    attribute_types = pd.read_csv(os.path.abspath("data/attribute_types.csv"), index_col=0)
    attribute_types["a_type"] = attribute_types["a_type"].apply(lambda x: literal_eval(x))

    df = df.merge(aliases, left_on="a_uri", right_on="uri", how="left")
    df = df.drop(columns=["uri"])
    df["possible_answers"] = df['aliases'].apply(lambda x: x if type(x) == list else [])
    df["possible_answers"] = df.progress_apply(lambda x: x["possible_answers"] + [x["attribute"]], axis=1)
    df = df.drop(columns=["aliases"])
    df = df.merge(attribute_types, left_on="a_uri", right_on="uri", how="left")
    df = df.drop(columns=["uri"])
    return df

def build_prompts(df, for_image=False):
    def best_obj_type(obj_types):
        prioritized_obj_types = ["city", "capital city", 'metropolis', 'country', 'occupation', 'language', 'type of sport', 'music genre'] # 'cinematic technique', 'team sport'
        for ot in prioritized_obj_types:
            if ot in obj_types:
                return ot
            for ot_ in obj_types:
                if "university" in ot_:
                    return "university"
                if "city" in ot_:
                    return "city"
        return obj_types[0]
    df = df.drop("subject", axis=1)
    subjects = pd.read_csv(os.path.abspath("data/meta.csv"))[["s_uri", "subject"]]
    df = df.merge(subjects, on=["s_uri"])
    templates = pd.read_csv(os.path.abspath("data/relation_templates.csv"))
    df = df.merge(templates[["uri", "template"]], left_on="r_uri", right_on="uri")
    df = df.drop(columns=["uri"])
    df = df.dropna()
    query_counts = df.drop_duplicates(["s_uri", "r_uri"]).groupby(["s_uri"])["r_uri"].count().reset_index(name="count")
    df = df.merge(query_counts[query_counts["count"] > 1][["s_uri"]], on="s_uri")

    df["question_for_image"] = df.progress_apply(lambda row: row["template"].replace("[subj]", "the subject of this image"), axis=1)
    df["question_for_image"] = df.progress_apply(lambda row: row["question_for_image"].replace("[obj_type]", best_obj_type(row["a_type"])) if len(row["a_type"]) > 0 else row["question"], axis=1)
    df["question"] = df.progress_apply(lambda row: row["template"].replace("[subj]", row["subject"]), axis=1)
    df["question"] = df.progress_apply(lambda row: row["question"].replace("[obj_type]", best_obj_type(row["a_type"])) if len(row["a_type"]) > 0 else row["question"], axis=1)
    df = df.drop(columns=["template"])
    df.to_csv("data/more_questions.csv", index=False)

build_prompts(aggregate_triples()) 