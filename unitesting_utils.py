import json
import urllib.request

def load_impact_theory_data():
    '''
    Loads impact_theory_data.json data by trying three options:
    1. Assumes user is in Google Colab environment and loads file from content dir.
    2. If 1st option doesn't work, assumes user is in course repo and loads from data dir.
    3. If 2nd option doesn't work, assumes user does not have direct access to data so
       downloads data direct from course repo.
    '''
    try:
        path = '/content/impact_theory_data.json'
        with open(path) as f:
            data = json.load(f)
        return data
    except Exception:
        print(f"Data not available at {path}")
        try: 
            path = './data/impact_theory_data.json'
            with open(path) as f:
                data = json.load(f)
            return data
        except Exception:
            print(f'Data not available at {path}, downloading from source')
            try:
                with urllib.request.urlopen("https://ra.githubusercontent.com/americanthinker/vectorsearch-applications/main/data/impact_theory_data.json") as url:
                    data = json.load(url)
                return data
            except Exception:
                print('Data cannot be loaded from source, please move data file to one of these paths to run this test:\n\
    1. "/content/impact_theory_data.json"   --> if you are in Google Colab\n\
    2. "./data/impact_theory_data.json"      --> if you are in a local environment\n')