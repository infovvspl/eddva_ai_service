import requests
import json

BASE_URL = "http://127.0.0.1:9000"

def test_endpoint(name, method, path, data=None, params=None):
    url = f"{BASE_URL}{path}"
    print(f"\n--- Testing {name} [{method}] ---")
    try:
        if method == "GET":
            resp = requests.get(url, params=params)
        else:
            resp = requests.post(url, json=data)
        
        if resp.status_code < 400:
            print(f"✅ Success ({resp.status_code})")
            # print(json.dumps(resp.json(), indent=2)) # Uncomment to see full response
        else:
            print(f"❌ Failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"🚨 Error: {e}")

if __name__ == "__main__":
    # 1. Root
    test_endpoint("Root API", "GET", "/")

    # 2. Feedback
    test_endpoint("Feedback Analyze", "POST", "/feedback/analyze/", data={
        "subject": "History",
        "student_answer": "The French Revolution started in 1789.",
        "marking_scheme": "Must mention the year 1789 and the causes."
    })

    # 3. Notes List
    test_endpoint("Notes List", "GET", "/notes/list/", params={"student_id": "student_001"})

    # 4. Content Suggest
    test_endpoint("Content Suggest", "GET", "/content/suggest/", params={"topic": "Quantum Computing"})

    # 5. Practice Test
    test_endpoint("Practice Test", "POST", "/test/generate/", data={
        "topic": "Python Functions",
        "num_questions": 3,
        "difficulty": "easy"
    })

    # 6. Career Roadmap
    test_endpoint("Career Roadmap", "POST", "/career/generate/", data={
        "goal": "Cloud Architect",
        "interests": ["AWS", "DevOps"]
    })

    # 7. Performance Analysis
    test_endpoint("Performance Analysis", "POST", "/performance/analyze/", data={
        "student_id": "stud_001",
        "subjects_data": [{"subject": "Math", "scores": [80, 90], "test_dates": ["2023-01-01", "2023-02-01"]}]
    })

    # 8. Personalization
    test_endpoint("Personalization", "POST", "/personalization/generate/", data={
        "student_id": "stud_001",
        "learning_style": "Auditory",
        "subjects_to_focus": ["English"],
        "available_hours_per_day": 2
    })

    # 9. Cheating Detection
    test_endpoint("Cheating Detection", "POST", "/cheating/analyze-logs/", data=[
        {"timestamp": "10:00:00", "event": "Tab Switch"},
        {"timestamp": "10:05:00", "event": "Camera Obscured"}
    ])

    print("\n--- Testing Complete ---")
