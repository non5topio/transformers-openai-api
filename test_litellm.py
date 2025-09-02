import litellm
import os

# Set up the configuration for your local API
os.environ['OPENAI_API_BASE'] = 'http://127.0.0.1:8001/v1'
os.environ['OPENAI_API_KEY'] = 'dummy-key-for-local-testing'  # Add this for local testing

def test_local_api():
    try:
        # Test with litellm
        response = litellm.completion(
            model="openai/qwen3-coder",  # This matches your env var format
            messages=[
                {"role": "user", "content": "Hello, just say hi and nothing else, just reply with Hi"}
            ],
            api_base="http://127.0.0.1:8000/v1",
            max_tokens=10,
            temperature=0.1
        )
        
        print("Success!")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    test_local_api()