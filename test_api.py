"""
API Testing Script for Voice Assistant Backend
Run this after starting the backend to verify all endpoints work
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_test(test_name):
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {test_name}")
    print(f"{'='*60}")

def print_response(response):
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_health_check():
    print_test("Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)
    return response.status_code == 200

def test_root():
    print_test("Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print_response(response)
    return response.status_code == 200

def test_send_chat():
    print_test("Send Chat Message")
    payload = {
        "message": "Hello, how are you?",
        "translate": False,
        "target_lang": "hi"
    }
    response = requests.post(f"{BASE_URL}/api/chat", json=payload)
    print_response(response)
    return response.status_code == 200

def test_send_translated_chat():
    print_test("Send Chat Message with Translation")
    payload = {
        "message": "Tell me about artificial intelligence",
        "translate": True,
        "target_lang": "hi"
    }
    response = requests.post(f"{BASE_URL}/api/chat", json=payload)
    print_response(response)
    return response.status_code == 200

def test_get_all_chats():
    print_test("Get All Chats")
    response = requests.get(f"{BASE_URL}/api/chats?limit=10")
    print_response(response)
    return response.status_code == 200

def test_get_stats():
    print_test("Get Statistics")
    response = requests.get(f"{BASE_URL}/api/stats")
    print_response(response)
    return response.status_code == 200

def test_delete_specific_chat():
    print_test("Delete Specific Chat")
    # First get all chats
    response = requests.get(f"{BASE_URL}/api/chats?limit=1")
    data = response.json()
    
    if data['chats']:
        chat_id = data['chats'][0]['id']
        print(f"Attempting to delete chat: {chat_id}")
        
        payload = {"chat_ids": [chat_id]}
        response = requests.delete(f"{BASE_URL}/api/chats/delete", json=payload)
        print_response(response)
        return response.status_code == 200
    else:
        print("No chats available to delete")
        return True

def test_initialize_chatbot():
    print_test("Initialize Chatbot")
    payload = {
        "whisper_model": "base",
        "llm_model": "llama3.1:8b",
        "glasses_device": None
    }
    response = requests.post(f"{BASE_URL}/api/initialize", json=payload)
    print_response(response)
    return response.status_code == 200

def run_all_tests():
    print("\n" + "="*60)
    print("🚀 Starting API Tests for Voice Assistant Backend")
    print("="*60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root),
        ("Initialize Chatbot", test_initialize_chatbot),
        ("Get Statistics", test_get_stats),
        ("Send Chat Message", test_send_chat),
        ("Send Translated Chat", test_send_translated_chat),
        ("Get All Chats", test_get_all_chats),
        ("Delete Specific Chat", test_delete_specific_chat),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            time.sleep(1)  # Small delay between tests
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*60}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*60}\n")
    
    return passed == total

if __name__ == "__main__":
    try:
        print("Checking if backend is running...")
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print("✅ Backend is running!\n")
        
        success = run_all_tests()
        
        if success:
            print("🎉 All tests passed! Your API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to backend!")
        print(f"Make sure the backend is running on {BASE_URL}")
        print("Start it with: python backend/api.py")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")