from groq import Groq
from colorama import Fore

client = Groq(api_key='YOUR_API_KEY')

def chat(prompt):
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are a personal healthcare assistant named HappyBot. You should ask user for diet and suggest any modifications and healthcare plans."
            },
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in completion:
        print(Fore.GREEN + str(chunk.choices[0].delta.content) or "", end="")

while True:
    x = input(Fore.CYAN + "\n\nUser : ")
    chat(x)
