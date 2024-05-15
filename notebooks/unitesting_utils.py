import json
import urllib.request
import os

def load_podcast_data(dataset_name: str='huberman_labs.json') -> list[dict]:
    '''
    Loads dataset in json format by trying three options:
    1. Assumes user is in Google Colab environment and loads file from content dir.
    2. If 1st option doesn't work, assumes user is in course notebooks repo and loads from data dir.
    3. If 2nd option doesn't work, assumes user does not have direct access to data so
       downloads data direct from Public course repo.
    '''
    try:
        path = f'/content/{dataset_name}'
        with open(path) as f:
            data = json.load(f)
        return data
    except Exception:
        print(f"Data not available at {path}")
        try: 
            path = f"{os.path.join(os.path.abspath('../'), f'data/{dataset_name}')}"
            with open(path) as f:
                data = json.load(f)
            return data
        except Exception:
            print(f'Data not available at {path}, downloading from source')
            try:
                with urllib.request.urlopen(f"https://raw.githubusercontent.com/americanthinker/vector_search_applications_PUBLIC/master/{dataset_name}") as url:
                    data = json.load(url)
                return data
            except Exception:
                print(f'Data cannot be loaded from source, please move data file to one of these paths to run this test:\n\
    1. "/content/{dataset_name}"   --> if you are in Google Colab\n\
    2. "../data/{dataset_name}"      --> if you are in a local environment\n')