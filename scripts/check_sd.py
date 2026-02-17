import requests
try:
    r = requests.get('http://127.0.0.1:7860/sdapi/v1/sd-models', timeout=3)
    models = r.json()
    print(f'SD Forge running! {len(models)} models')
    for m in models[:5]:
        print(f'  {m["title"]}')
except Exception as e:
    print(f'SD Forge not running: {e}')
