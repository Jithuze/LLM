import os
import chromadb
from PyPDF2 import PdfReader
from colorama import Fore, init
import ollama

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Initialize ChromaDB client
client = chromadb.Client()

# Define the PDF folder
PDF_FOLDER = 'my_pdf'

# Create vector database
def create_vector_db():
    vector_db_name = 'pdf_collection'

    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    return client.create_collection(name=vector_db_name)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Function to process the PDF folder and add/update vector database
def process_pdf_folder(vector_db):
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

    if not pdf_files:
        print(Fore.RED + "No PDF files found in the folder.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(Fore.YELLOW + f"Processing PDF: {pdf_file}...")

        pdf_content = extract_text_from_pdf(pdf_path)

        # Generate embeddings for the PDF content
        response = ollama.embeddings(model='nomic-embed-text', prompt=pdf_content)
        embedding = response['embedding']

        # Add or update the document in the vector database
        vector_db.add(
            ids=[pdf_file],
            embeddings=[embedding],
            documents=[pdf_content]
        )
        print(Fore.GREEN + f"Successfully processed and added to the database: {pdf_file}")

# Function to retrieve similar content based on user query
def retrieve_content_from_db(vector_db, query):
    response = ollama.embeddings(model='nomic-embed-text', prompt=query)
    query_embedding = response['embedding']

    result = vector_db.query(query_embeddings=[query_embedding], n_results=3)
    if result['documents']:
        best_matches = [doc for doc in result['documents'][0]]
        return " ".join(best_matches)  # Return concatenated relevant documents
    else:
        print(Fore.RED + "No results found for your query.")
        return None

# Function to generate a response based on user query and retrieved content
def generate_response(query, retrieved_content):
    if retrieved_content:
        prompt = f"Answer the question based on the following information: {retrieved_content}\nQuestion: {query}"
    else:
        prompt = f"Question: {query}"

    response = ''
    stream = ollama.chat(model='phi3.5', messages=[{'role': 'user', 'content': prompt}], stream=True)
    
    print(Fore.GREEN + "\nMODEL RESPONSE: ")
    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(Fore.GREEN + content, end='', flush=True)
    
    print('\n')
    return response

# Function for terminal user interaction menu
def user_interaction_menu(vector_db):
    while True:
        print(Fore.CYAN + "\n--- Menu ---")
        print(Fore.BLUE + "1. Process PDF folder and add to vector database")
        print(Fore.BLUE + "2. Ask a question and get a response")
        print(Fore.BLUE + "3. Exit")
        
        choice = input(Fore.CYAN + "Enter your choice (1/2/3): ")

        if choice == '1':
            print(Fore.YELLOW + "Processing PDF folder...")
            process_pdf_folder(vector_db)
        elif choice == '2':
            query = input(Fore.CYAN + "Enter your question: ")
            retrieved_content = retrieve_content_from_db(vector_db, query)
            generate_response(query, retrieved_content)
        elif choice == '3':
            print(Fore.MAGENTA + "Exiting the program... Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid choice. Please try again.")

# Main execution
if __name__ == "__main__":
    print(Fore.GREEN + "Welcome to the PDF Knowledge Assistant!")
    vector_db = create_vector_db()
    user_interaction_menu(vector_db)
