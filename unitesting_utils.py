import json
import urllib.request

def load_impact_theory_data():
    try:
        path = './content/impact_theory_data.json'
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
                with urllib.request.urlopen("https://raw.githubusercontent.com/americanthinker/vectorsearch-applications/main/data/impact_theory_data.json") as url:
                    data = json.load(url)
                return data
            except Exception:
                print('Data cannot be loaded from source, please move data file to one of the above paths to run this test')