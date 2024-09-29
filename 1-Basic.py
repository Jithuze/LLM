import ollama

while True:
    prompt = input('USER : ')
    convo = []
    convo.append({'role': 'user', 'content': prompt})

    output = ollama.chat(model='phi3.5', messages=convo)

    print(f"MODEL : {output['message']['content']}")
    print("\n")


#    THIS AGENT DOES NOT HAVE CONVERSATIONAL MEMMORY