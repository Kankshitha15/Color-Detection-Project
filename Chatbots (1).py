#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import openai
import re
import random
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )
        message = response.choices[0].text.strip()
        return message
    except Exception as e:
        print(f"Error: {e}")
        return "I'm sorry, I couldn't generate a response. Please try again later."

def get_bot_response(user_input):
    # Apply basic preprocessing to the user input
    user_input = user_input.lower().strip()

    # Check for common greeting patterns
    greeting_patterns = [r'hi|hello|hey', r'good (morning|afternoon|evening)']
    for pattern in greeting_patterns:
        if re.search(pattern, user_input):
            return random.choice(["Hello!", "Hi there!", "Hey, how's it going?"])

    # Check for common goodbye patterns
    goodbye_patterns = [r'bye|goodbye|see you', r'have a good (day|night)']
    for pattern in goodbye_patterns:
        if re.search(pattern, user_input):
            return random.choice(["Goodbye!", "See you later!", "Have a great day!"])

    # Generate a response using the OpenAI GPT-3 API
    prompt = f"User: {user_input}\nAssistant: "
    response = generate_response(prompt)
    return response

# Main chatbot loop
while True:
    user_input = input("You: ")

    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break

    bot_response = get_bot_response(user_input)
    print(f"Chatbot: {bot_response}")

