import re
import os
import pandas as pd
import ast

import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
from urllib.parse import urlparse

from utils import load_json_df_from_dir


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
        self.sparql.setQuery(query)
        responses = self.sparql.query().convert()

        records = []
        for response in responses['results']['bindings']:
            record = {}
            for key in response:
                record[key] = self.parse_value(response[key]['value'])
            records.append(record)
        return pd.DataFrame(records)

# Step -1
def relation_aliases():
    def _query(relation_ids): 
        return f'''
SELECT ?relation_id ?alias
WHERE {{
    {{VALUES ?relation_id {{ {" ".join([f"wd:{id}" for id in relation_ids])} }} }}
    ?relation_id skos:altLabel ?alias.
    FILTER(LANG(?alias) = "en")
}}
'''
    counterfact_df = pd.read_json("samples/counterfact.json")
    counterfact_df["relation_id"] = counterfact_df["requested_rewrite"].apply(lambda x: x["relation_id"])
    relation_ids = counterfact_df["relation_id"].drop_duplicates().to_list()

    sparql = SPARQL()
    df = sparql.execute(_query(relation_ids))
    df.to_csv("data/relation_aliases.csv")

def relation_labels():
    def _query(relation_ids): 
        return f'''
SELECT ?relation_id ?relation
WHERE {{
    {{VALUES ?relation_id {{ {" ".join([f"wd:{id}" for id in relation_ids])} }} }}
    ?relation_id rdfs:label ?relation.  filter(lang(?relation) = "en").
}}
'''
    df = pd.read_csv("data/relation_aliases.csv", index_col=0)
    relation_ids = df["relation_id"].drop_duplicates().to_list()

    sparql = SPARQL()
    df2 = sparql.execute(_query(relation_ids))
    df3 = pd.read_csv("data/relation_aliases.csv", index_col=0)
    df = pd.concat([df3, df2.rename(columns={"relation": "alias"})])
    df = df.drop_duplicates()
    df.to_csv("data/relation_aliases.csv")
    print("here")


# Step 0
def extract_examples():
    # df = pd.read_json("data/old_data/mistakes_no_good_relation_before_subject_pred_attribute_rank.json")
    df = load_json_df_from_dir("data/old_data/good_prompts/knowns_relation_before_subject_good_prompts")
    df = df[['subject', 'attribute', 'relation_id']]
    df = df.drop_duplicates()
    df = df.reset_index().drop("index", axis=1)

    counterfact_df = pd.read_json("data/counterfact.json")
    counterfact_df["relation_id"] = counterfact_df["requested_rewrite"].apply(lambda x: x["relation_id"])
    counterfact_df["attribute"] = counterfact_df["requested_rewrite"].apply(lambda x: " " + x["target_true"]["str"])
    counterfact_df["attribute_id"] = counterfact_df["requested_rewrite"].apply(lambda x: x["target_true"]["id"])
    counterfact_df = counterfact_df[["relation_id", "attribute", "attribute_id"]]
    counterfact_df = counterfact_df.drop_duplicates()

    df = df.merge(counterfact_df, on=["relation_id", "attribute"])
    # df.to_csv("data/step_0/mistakes_sample.csv")
    df.to_csv("data/step_0/knowns_sample.csv")


# Step 1
def attribute_type():
    def _query(attribute_ids):
        return f'''
SELECT ?attribute_id ?typeLabel
WHERE {{
  {{VALUES ?attribute_id {{ {" ".join([f"wd:{id}" for id in attribute_ids])} }} }}
  ?attribute_id wdt:P31 ?type.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}        
'''
    # df = pd.read_csv("data/step_0/mistakes_sample.csv", index_col=0) # TODO(daniela): this should be a parameter
    df = pd.read_csv("data/step_0/knowns_sample.csv", index_col=0)
    attribute_ids = df["attribute_id"].drop_duplicates().to_list()

    sparql = SPARQL()
    df2 = sparql.execute(_query(attribute_ids))
    df = df.merge(df2, on="attribute_id")
    df = df.rename(columns={"typeLabel": "attribute_type"})
    # df.to_csv("data/step_1/mistakes_sample.csv")
    df.to_csv("data/step_1/knowns_sample.csv")


# Step 2 
def subject_type():
    def _query(subjects):
        subjects = [subject.replace("'", "\\'") for subject in subjects]
        return f'''
SELECT DISTINCT ?subject ?subject_id
WHERE {{
  VALUES ?subject {{ {" ".join([f"'{subject}'" for subject in subjects])} }}
  ?uri ?predicate ?subject .
  BIND(STRBEFORE(STR(?uri), "-") AS ?subject_id)
  FILTER(?predicate = pq:P1810 && ?subject_id != "")
}}
'''
    # df = pd.read_csv("/home/gamir/DER-Roei/dhgottesman/hallucasting/data/step_0/mistakes_sample.csv")
    df = pd.read_csv("/home/gamir/DER-Roei/dhgottesman/hallucasting/data/step_0/knowns_sample.csv")

    df = df[['subject', 'attribute_id', 'relation_id']]
    df = df.drop_duplicates()
    df = df.reset_index().drop("index", axis=1)

    subjects = df["subject"].drop_duplicates().to_list()
    subject_chunks = [subjects[i:i+20] for i in range(0, len(subjects), 20)]
    sparql = SPARQL()
    df = df.merge(pd.concat([sparql.execute(_query(chunk)) for chunk in subject_chunks]), on="subject")

    def _query(triples):
        return f'''
SELECT DISTINCT ?subject_id ?relation_id ?attribute_id
WHERE {{
  VALUES (?subject_id ?relation_id ?attribute_id) {{ {" ".join([f"(wd:{subject_id} wdt:{relation_id} wd:{attribute_id})" for subject_id, relation_id, attribute_id in triples])} }}
  ?subject_id ?relation_id ?attribute_id .
}}       
'''
    triples = [tuple(x) for x in df[['subject_id', 'relation_id', 'attribute_id']].itertuples(index=False)]
    triples_chunks = [triples[i:i+20] for i in range(0, len(triples), 20)]
    sparql = SPARQL()
    df2 = df.merge(pd.concat([sparql.execute(_query(chunk)) for chunk in triples_chunks]), on=['subject_id', 'attribute_id', 'relation_id'])

    def _query(subject_ids):
        return f'''
SELECT ?subject_id ?typeLabel
WHERE {{
  {{VALUES ?subject_id {{ {" ".join([f"wd:{id}" for id in subject_ids])} }} }}
  ?subject_id wdt:P31 ?type.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}        
'''
    subject_ids = df2["subject_id"].drop_duplicates().to_list()
    df3 = sparql.execute(_query(subject_ids))
    df4 = df2.merge(df3, on="subject_id")
    df4 = df4.rename(columns={"typeLabel": "subject_type"})
    # df4.to_csv("data/step_2/mistakes_sample.csv")
    df4.to_csv("data/step_2/knowns_sample.csv")


# Step 3
def input_to_template():
    # df = pd.read_csv("data/step_1/mistakes_sample.csv", index_col=0) # TODO(daniela): this should be a parameter
    df = pd.read_csv("data/step_1/knowns_sample.csv", index_col=0) # TODO(daniela): this should be a parameter

    df2 = pd.read_csv("data/relation_aliases.csv", index_col=0)

    df = df.merge(df2, on="relation_id")
    df = df.rename(columns={"alias": "relation_alias"})
    df = df.groupby(['subject', 'attribute', 'relation_id', 'attribute_id']).agg({
        'attribute_type': lambda x: list(set(x)),
        'relation_alias': lambda x: list(set(x))
    }).reset_index()

    # df3 = pd.read_csv("data/step_2/mistakes_sample.csv", index_col=0)
    df3 = pd.read_csv("data/step_2/knowns_sample.csv", index_col=0)
    df3 = df3.groupby(['subject', 'relation_id', 'attribute_id', 'subject_id']).agg({
        'subject_type': lambda x: list(set(x)),
    }).reset_index()
    df4 = df.merge(df3, on=["subject", "relation_id", "attribute_id"])
    df4.to_csv("data/step_3/knowns_sample.csv")
    # df4.to_csv("data/step_3/mistakes_sample.csv")


if __name__ == '__main__':
    input_to_template()
