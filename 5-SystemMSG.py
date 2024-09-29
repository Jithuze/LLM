import chromadb
import ollama
from colorama import Fore


sys_msg = "You are an assistant who is expert in giving advice on mobile phones. You are an AI assistant that has memory about every conversation you have with the user. Give only the relevant information and keep the conversation short and to the point. Also respond normally is the user message does not have mobile phone in the context."


convo = [{'role': 'system', 'content': sys_msg}]
client = chromadb.Client()

message_history = [
  {
    "id": 1,
    "prompt": "Hey, what are the latest mobile phones in the market?",
    "response": "Hi! Some of the latest mobile phones include the iPhone 15, Samsung Galaxy S23, Google Pixel 8, and OnePlus 12. These phones have some advanced features like improved camera systems, faster processors, and more efficient battery life. Are you looking for something specific?"
  },
  {
    "id": 2,
    "prompt": "What’s special about the iPhone 15?",
    "response": "The iPhone 15 introduces several key features like a Dynamic Island, a 48MP main camera, and an A16 Bionic chip for faster performance. It also has USB-C charging, a slightly larger battery, and some cool new color options. It's a great upgrade if you're into photography or gaming!"
  },
  {
    "id": 3,
    "prompt": "Does the iPhone 15 have a better battery than the previous models?",
    "response": "Yes, the iPhone 15 has a slightly improved battery life compared to the iPhone 14. Apple has focused on optimizing energy efficiency with the A16 chip, so you should notice better longevity, especially with regular day-to-day use. It should last about 1-2 hours longer depending on usage."
  },
  {
    "id": 4,
    "prompt": "How does the Samsung Galaxy S23 compare to the iPhone 15?",
    "response": "The Galaxy S23 and iPhone 15 are both premium phones, but they cater to slightly different audiences. The S23 has a Snapdragon 8 Gen 2 chip, a powerful 200MP camera on the Ultra model, and Samsung’s signature AMOLED display with a 120Hz refresh rate. In contrast, the iPhone 15 focuses on iOS integration, ease of use, and camera quality. It really depends on whether you prefer Android or iOS."
  },
  {
    "id": 5,
    "prompt": "Which one has a better camera, the iPhone 15 or the Samsung S23?",
    "response": "Both have excellent cameras, but the Samsung Galaxy S23 Ultra arguably has the edge with its 200MP main sensor and 10x optical zoom, making it great for long-range shots and highly detailed photos. The iPhone 15’s 48MP camera is impressive for photography too, especially with Apple’s computational photography, but Samsung’s zoom capabilities outshine it if that’s important to you."
  },
  {
    "id": 6,
    "prompt": "What’s new with Google Pixel 8?",
    "response": "The Google Pixel 8 brings improved AI-driven photography features, such as Magic Editor and Best Take, along with Google's custom-built Tensor G3 chip for faster performance. The display now supports a 120Hz refresh rate, and the phone also has enhanced battery efficiency. The Pixel 8’s software experience with pure Android 14 is also a big draw for many users."
  },
  {
    "id": 7,
    "prompt": "Does the Pixel 8 support 5G?",
    "response": "Yes, the Google Pixel 8 fully supports 5G, both mmWave and sub-6GHz bands, ensuring fast download speeds and better connectivity in areas with 5G coverage."
  },
  {
    "id": 8,
    "prompt": "What makes the OnePlus 12 stand out?",
    "response": "The OnePlus 12 stands out with its fluid 120Hz AMOLED display, Snapdragon 8 Gen 3 processor, and ultra-fast 150W charging, which can charge the phone in about 15 minutes. It also features a versatile camera system developed in partnership with Hasselblad, ensuring stunning photography and color accuracy."
  },
  {
    "id": 9,
    "prompt": "How does the charging speed of OnePlus 12 compare to the iPhone 15?",
    "response": "The OnePlus 12 charges much faster than the iPhone 15. With its 150W charging, it can reach full battery in as little as 15-20 minutes, whereas the iPhone 15, even with its USB-C upgrade, still charges more slowly, typically taking around an hour to reach full charge with fast charging."
  },
  {
    "id": 10,
    "prompt": "Which one would you recommend overall?",
    "response": "It depends on what you value most in a phone. If you prefer a seamless iOS experience, great performance, and top-notch security, the iPhone 15 is a great choice. If you want the best camera and display, the Samsung Galaxy S23 Ultra is hard to beat. For AI-driven photography and a clean Android experience, go with the Google Pixel 8. The OnePlus 12 is perfect if you want fast charging and a high-performance device at a slightly lower price."
  }
]


def stream_response(prompt):
    convo.append({'role': 'user', 'content': prompt})
    response = ''
    stream = ollama.chat(model='phi3.5', messages=convo, stream=True)
    print(Fore.GREEN + "\nMODEL: ")

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(Fore.GREEN + content, end='', flush=True)
    
    convo.append({'role': 'assistant', 'content': response})
    print('\n')


def create_vector_db(conversations):
    vector_db_name = 'conversations'

    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    vector_db = client.create_collection(name=vector_db_name)

    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"
        #Pull model using 'ollama pull nomin-embed-text'
        response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)     
        embedding = response['embedding']

        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )

def retrieve_embeddings(prompt):
    response = ollama.embeddings(model='nomic-embed-text', prompt=prompt) 
    prompt_embedding = response['embedding']

    vector_db = client.get_collection(name='conversations')
    result = vector_db.query(query_embeddings=[prompt_embedding],n_results=1)

    # print(f"Results From Vector Search : {result}")
    best_embedding = result['documents'][0][0]

    return best_embedding

create_vector_db(conversations=message_history)

while True:
    prompt = input(Fore.CYAN + 'USER : ')
   
    context = retrieve_embeddings(prompt=prompt)
    prompt = f"USER PROMPT : {prompt} \nCONTEXT FROM EMBEDDING : {context}"
    stream_response(prompt)
    print("\n")