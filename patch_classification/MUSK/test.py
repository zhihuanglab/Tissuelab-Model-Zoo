import requests
import json

url = "http://localhost:5001/api/tasks/v1/start_workflow"

payload = {
    "h5_path": "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\TCGA-A6-6648-01Z-00-DX1.88b9a490-0bed-43f3-bd74-1bf2810f6884.svs.h5",
    "step1": {
        "model": "MuskNode",
        "input": {
            "path": "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\TCGA-A6-6648-01Z-00-DX1.88b9a490-0bed-43f3-bd74-1bf2810f6884.svs",
            "classifier_path": "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\classifier.xgb",
            "save_classifier_path": "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\classifier_1.xgb",
            "tissue_classes": [
                "Negative control",
                "Neurons",
                "Tumor",
                "Microglia"
            ],
            "tissue_colors": [
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