## Build a simple OCR API service by modelscope

### Introduction

1. The core code comes from：https://modelscope.cn/headlines/article/42
2. Resolve dependency package conflicts
3. Provide a simple HTTP API

### Usage

1. Run OCR API service
```shell
conda create -n ocr python=3.11
conda activate ocr
pip install -r requirements.txt
python main.py
```

2. Call OCR API on the client, for example:
```python
import requests

def ocr(image_path):
    files = {'file': open(image_path, 'rb')}
    response = requests.post(f'http://localhost:7700/api/ocr', files=files)
    if response.status_code == 200:
        response_json = response.json()
        text = response_json["data"]
        return text
    return None

if __name__ == '__main__':
    text = ocr('/Users/valdanito/Downloads/test222.jpg')
    print(text)

```

