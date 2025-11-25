"""
Test script to verify GPT-5.1 API access.
"""
import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set")
    exit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY)

print("Testing GPT-5.1 access...")
print("=" * 60)

# Test 1: List available models
print("\n1. Checking available models...")
try:
    models = client.models.list()
    gpt5_models = [m.id for m in models.data if 'gpt-5' in m.id.lower()]
    if gpt5_models:
        print(f"✓ Found GPT-5 models: {gpt5_models}")
    else:
        print("✗ No GPT-5 models found in your account")
        print("Available models include:", [m.id for m in models.data][:10])
except Exception as e:
    print(f"✗ Error listing models: {e}")

# Test 2: Try Responses API with GPT-5.1
print("\n2. Testing Responses API with gpt-5.1...")
try:
    resp = client.responses.create(
        model="gpt-5.1",
        input="Say 'Hello, GPT-5.1 works!'",
        reasoning={"effort": "low"},
        text={"verbosity": "low"}
    )
    print(f"✓ GPT-5.1 works! Response: {resp.output_text}")
except Exception as e:
    print(f"✗ GPT-5.1 failed: {str(e)[:200]}")

# Test 3: Try GPT-5-mini
print("\n3. Testing Responses API with gpt-5-mini...")
try:
    resp = client.responses.create(
        model="gpt-5-mini",
        input="Say 'Hello, GPT-5-mini works!'",
        reasoning={"effort": "low"},
        text={"verbosity": "low"}
    )
    print(f"✓ GPT-5-mini works! Response: {resp.output_text}")
except Exception as e:
    print(f"✗ GPT-5-mini failed: {str(e)[:200]}")

print("\n" + "=" * 60)
print("Test complete.")
