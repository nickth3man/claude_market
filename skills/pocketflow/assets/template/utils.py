"""
PocketFlow Template - Utility Functions

Source: https://github.com/The-Pocket/PocketFlow-Template-Python

This module contains utility functions like LLM wrappers.
"""

import os


def call_llm(prompt):
    """
    Call your LLM provider

    Args:
        prompt (str): The prompt to send to the LLM

    Returns:
        str: The LLM response

    TODO: Implement your LLM provider here
    """

    # Example: OpenAI
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
    """

    # Example: Anthropic
    """
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-0",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
    """

    # Example: Google Gemini
    """
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=prompt
    )
    return response.text
    """

    raise NotImplementedError(
        "Implement your LLM provider in utils.py\n"
        "See examples above for OpenAI, Anthropic, or Google Gemini"
    )
