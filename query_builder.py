import ast
import ollama
from colorama import Fore


def create_queries(prompt):
    query_msg = (
    "You are a first-principles reasoning search query AI agent. "
    "Given the user's prompt, generate a Python list of short search queries "
    "that would retrieve any relevant information from the embedding database "
    "of all conversations you have ever had with this user. "
    "Your output must be ONLY a valid Python list of strings, with no explanations, "
    "no extra text, and no syntax errors. "
    "Think in terms of concepts, facts, or context necessary to respond accurately."
    )

    query_convo = [
        {"role": "system", "content": query_msg},

        # Example 1: insurance
        {"role": "user", "content": "Write an email to my car insurance company and create a persuasive request for them to lower prices based on my good driving record"},
        {"role": "assistant", "content": '["User name", "Current auto insurance provider", "Driving record", "Insurance policy details"]'},

        # Example 2: Python voice assistant
        {"role": "user", "content": "How can I convert the speak function in my Llama3 Python voice assistant to use pyttsx3 instead?"},
        {"role": "assistant", "content": '["Llama3 voice assistant", "Python TTS libraries", "pyttsx3 usage", "Convert text-to-speech function"]'},

        # Example 3: cats / personal memory
        {"role": "user", "content": "Do you remember my cat Mellow?"},
        {"role": "assistant", "content": '["User cat name", "Previously mentioned pets", "Feline companion facts"]'},

        # Example 4: cooking / recipe
        {"role": "user", "content": "I want to make a chocolate cake from scratch, how should I do it?"},
        {"role": "assistant", "content": '["Chocolate cake recipes", "Baking instructions", "Ingredients for chocolate cake", "Cooking steps"]'},

        # Example 5: travel / planning
        {"role": "user", "content": "Can you suggest a travel itinerary for a week in Japan?"},
        {"role": "assistant", "content": '["Japan travel itinerary", "Top tourist spots in Japan", "Local customs and culture", "Recommended activities in Japan"]'},

        # Dynamic user prompt
        {"role": "user", "content": prompt},
    ]

    response = ollama.chat(model='llama3', messages=query_convo)
    print(Fore.LIGHTYELLOW_EX + f'\nVector database queries: {response["message"]["content"]}')
    try:
        return ast.literal_eval(response['message']['content'])
    except Exception:
        return [prompt]
