import requests
import time


def check_health():
    base_url = "http://127.0.0.1:8000"
    endpoints = [
        "/",
        "/feedback/health",
        "/notes/health",
        "/content/health",
        "/test/health",
        "/career/health",
        "/performance/health",
        "/personalization/health",
        "/cheating/health",
    ]

    print(f"Waiting for server to be ready at {base_url}...")
    for _ in range(30):
        try:
            resp = requests.get(base_url)
            if resp.status_code == 200:
                print("Server is UP!")
                break
        except:
            time.sleep(5)
    else:
        print("Server timed out.")
        return

    for ep in endpoints:
        try:
            resp = requests.get(base_url + ep)
            print(f"[{ep}] {resp.status_code} - {resp.json()}")
        except Exception as e:
            print(f"[{ep}] FAILED: {e}")


if __name__ == "__main__":
    check_health()
