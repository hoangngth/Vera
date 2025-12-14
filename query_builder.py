import ast
import ollama
from colorama import Fore


def create_queries(prompt):
    query_msg = (
        'You are a first principle reasoning search query AI agent. '
        'Your list of search queries will be ran on an embedding database of all your conversations '
        'you have ever had with the user. With first principles create a Python list of queries to '
        'search the embeddings database for any data that would be necessary to have access to in '
        'order to correctly respond to the prompt. Your response must be a Python list with no syntax errors. '
        'Do not explain anything and do not ever generate anything but a perfect syntax Python list'
    )

    query_convo = [
        {'role': 'system', 'content': query_msg},
        {'role': 'user', 'content': 'Write an email to my car insurance company and create a pursuasive request for them to lower prices based on my good driving record'},
        {'role': 'assistant', 'content': '["What is the users name?", "What is the users current auto insurance provider?", "What is the users driving record?"]'},
        {'role': 'user', 'content': 'how can i convert the speak function in my llama3 python voice assistant to use pyttsx3 instead'},
        {'role': 'assistant', 'content': '["Llama3 voice assistant", "Python voice assistant", "openAI TTS", "openai speak", "text to speech python", "convert TTS to pyttsx3"]'},
        {'role': 'user', 'content': prompt},
    ]

    response = ollama.chat(model='llama3', messages=query_convo)
    print(Fore.LIGHTYELLOW_EX + f'\nVector database queries: {response["message"]["content"]}')
    try:
        return ast.literal_eval(response['message']['content'])
    except Exception:
        return [prompt]
