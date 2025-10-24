#!/usr/bin/env python3
"""
Quick script to test your LLM connection
"""

import os
import sys

def test_openai():
    """Test OpenAI connection"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello'"}]
        )
        print("✅ OpenAI: Connected")
        print(f"   Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ OpenAI: Failed - {e}")
        return False

def test_anthropic():
    """Test Anthropic connection"""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'hello'"}]
        )
        print("✅ Anthropic: Connected")
        print(f"   Response: {response.content[0].text}")
        return True
    except Exception as e:
        print(f"❌ Anthropic: Failed - {e}")
        return False

def test_google():
    """Test Google Gemini connection"""
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents="Say 'hello'"
        )
        print("✅ Google Gemini: Connected")
        print(f"   Response: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Google Gemini: Failed - {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing LLM connections...\n")

    results = {
        "OpenAI": test_openai(),
        "Anthropic": test_anthropic(),
        "Google": test_google()
    }

    print("\n" + "="*50)
    working = [k for k, v in results.items() if v]
    if working:
        print(f"✅ Working providers: {', '.join(working)}")
    else:
        print("❌ No working providers found")
        print("\nMake sure you've set environment variables:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export GEMINI_API_KEY=...")
