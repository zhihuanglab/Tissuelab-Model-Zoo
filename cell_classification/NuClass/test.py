import requests
import json

url = "http://localhost:5001/api/tasks/v1/start_workflow"

payload = {
    "h5_path": "C:\\Users\\lsoho\\Git\\penn\\TissueLab\\example_WSI\\H&E\\CMU-1.svs.h5",
    "step1": {
        "model": "ClassificationNode",
        "input": {
            "path": "C:\\Users\\lsoho\\Git\\penn\\TissueLab\\example_WSI\\H&E\\CMU-1.svs",
            "classifier_path": "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\classifier.xgb",
            "save_classifier_path": "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\classifier_1.xgb",
            "nuclei_classes": [
                "Negative control",
                "Neurons",
                "Tumor",
                "Microglia"
            ],
            "nuclei_colors": [
                "#aaaaaa",
                "#7012e2",
                "#de1212",
                "#1be90c"
            ]
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