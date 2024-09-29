import ollama

convo = []

while True:
    prompt = input('USER : ')
   
    convo.append({'role': 'user', 'content': prompt})

    output = ollama.chat(model='phi3.5', messages=convo)

    response = output['message']['content']
    print(f"MODEL : {response}")
    
    convo.append({'role': 'assistant', 'content': response})
    print("\n")