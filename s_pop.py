import requests
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def get_entity_title(entity_id):
    base_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "props": "labels"
    }

    headers = {
        "User-Agent": "'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'"
    }

    response = requests.get(base_url, params=params, headers=headers)
    try:
        if response.status_code == 200:
            data = response.json()
            if entity_id in data["entities"] and "labels" in data["entities"][entity_id]:
                title = data["entities"][entity_id]["labels"]["en"]["value"]  # Assuming you want the English label
                title = title.replace(" ", "_")
                return title
            else:
                print("Title not found.")
                return None
        else:
            raise Exception("Bad response")
    except Exception:
        print("Failed to retrieve title.")
        return None

def get_entity_pageviews(title):
    base_url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    endpoint = f"en.wikipedia/all-access/user/{title}/monthly/20000101/20230131"  # Change dates as needed
    url = base_url + endpoint

    headers = {
        "User-Agent": "'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'"
    }

    response = requests.get(url, headers=headers)
    try:
        if response.status_code == 200:
            data = response.json()
            total_views = sum(data["items"][i]["views"] for i in range(len(data["items"])))
            return total_views
        else:
            raise Exception("Bad response")
    except Exception:
        print("Failed to retrieve pageviews.")
        return None


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("meta.csv")
    df = df.drop_duplicates(subset=["s_uri"])

    df["s_wiki_title"] = df["s_uri"].progress_apply(lambda x: get_entity_title(x))
    df["s_pop"] = df["s_wiki_title"].progress_apply(lambda x: get_entity_pageviews(x))
    df.to_csv("meta.csv")

