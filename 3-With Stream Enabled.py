import ollama

convo = []

def stream_response(prompt):
    convo.append({'role': 'user', 'content': prompt})
    response = ''
    stream = ollama.chat(model='phi3.5', messages=convo, stream=True)
    print("\nMODEL: ")

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)
    
    print('\n')

while True:
    prompt = input('USER : ')
   
    response = stream_response(prompt)
    
    convo.append({'role': 'assistant', 'content': response})
    print("\n")