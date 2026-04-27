import requests
import json

base_url = "http://localhost:8000" # Django port
headers = {
    "X-API-Key": "apexiq-dev-secret-key-2026",
    "Content-Type": "application/json"
}

def test_generate(qtype, topic="Newton Laws"):
    print(f"\nTesting type: {qtype}...")
    payload = {
        "topic": topic,
        "num_questions": 2,
        "difficulty": "medium",
        "type": qtype
    }
    try:
        resp = requests.post(f"{base_url}/test/generate/", json=payload, headers=headers, timeout=40)
        if resp.status_code == 200:
            data = resp.json()
            print(f"SUCCESS: Returned {len(data.get('questions', []))} questions")
            for q in data.get('questions', []):
                print(f" - Q: {q.get('question')[:60]}...")
                print(f"   A: {q.get('answer')}")
                if 'options' in q:
                    print(f"   Opts: {q.get('options')}")
        else:
            print(f"FAILED: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    test_generate("mcq")
    test_generate("integer")
    test_generate("mcq_multi")
    test_generate("match_the_following")
