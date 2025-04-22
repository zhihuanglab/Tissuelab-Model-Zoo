import requests
import json

url = "http://localhost:5001/api/tasks/v1/start_workflow"

payload = {
    "h5_path": "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\CMU-1.svs.h5",
    "step1": {
        "model": "MuskNode",
        "input": {
            "path": "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\CMU-1.svs",
            # "classifier_path": "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\classifier_params.h5",
            # "save_classifier_path": "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\classifier_params_new.h5",
            "tissue_classes": ["Negative control", "Epithelial", "Lymphocytes"],
            "tissue_colors": ["#aaaaaa", "#de1212", "#6311df"]
        }
    }
}

try:
    response = requests.post(url, json=payload)
    
    response.raise_for_status()
    
    print("response.status_code:", response.status_code)
    print("response.json():")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    
except requests.exceptions.RequestException as e:
    print(f"RequestException: {e}")