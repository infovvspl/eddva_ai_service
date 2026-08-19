import requests

try:
    url = "https://pub-22354a2b0e694b93bcce0d5fb28e22a2.r2.dev"
    print(f"Testing GET to {url}...")
    resp = requests.get(url, timeout=10)
    print("Status code:", resp.status_code)
    print("Headers:", resp.headers)
    print("Preview content:", resp.content[:100])
except Exception as e:
    print("Error:", e)
