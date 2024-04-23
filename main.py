#This code will use ollama3 to answer question about the card game UNO and its rules.

#First we read the pdf documents of the rules downloaded and save their text, the word used is "load data"
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

def load_documents():
    document_loader=PyPDFDirectoryLoader("data") #the attribute of PyPDFDirectoryLoader is the path of the folder with pdf data
    return document_loader.load()

documents = load_documents()
# now each pdf is saved inside document, a list that contains for each index a page of the pdf and its metadata
#print(documents[0])

#it is time to split the text in chunks and save them into a databe in their embedded version (multidimensional coordinates)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document 

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function= len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
#the function split_documents takes the loaded list as argunement, so it contains the text of the pages of the pdf an its realtive metadata,
#the chunk contains the splitted part of the page_content and its metadata, it is still a list

chunks = split_documents(documents)
#print(chunks[0])

#how to look inside chunks:
for chunk in chunks:
    source = chunk.metadata.get("source")
    page = chunk.metadata.get("page")
    current_page_id = f"{source}:{page}"
    #let's add a unique id for each chunk made of the extractted metadata, 
    #it will be useful to add and remove chunks from the database
    chunk.metadata["id"] = current_page_id
    #print(current_page_id) the print let you look the metadata content of each iteration 

#now is time to define an embedding function for the chunks, remember to use the same embedding to store and to retrive the data otherwise it wont work
from langchain_community.embeddings.ollama import OllamaEmbeddings

# i have switched the ollama port to localhost to 8888 and pulled the model for embedding nomic-embed-text
# OLLAMA_HOST=0.0.0.0:8888 ollama pull nomic-embed-text

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

#now we create the vector database with CHROMADB (openAI)
from langchain.vectorstores.chroma import Chroma

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory="chroma", embedding_function=get_embedding_function() #the persist directory needs the path for the db, i will use the folder chroma 
    )
    db.add_documents(chunks)#,ids=id creato per i chunk, altrimenti quello autoamticamente generato non può essere ustao per riconoscere i record già scritti)
    db.persist()
    return db

db = add_to_chroma(chunks) #aggiunge i chunk al db nella cartella chroma si genera il file db sqllite3


#now we define the prompt schema that will give the right context to the llm before answering

from langchain_core.prompts import ChatPromptTemplate
#load the embedding funtion and chroma db if you write this in another file, so that the chunk embedded can be seen
def query_rag(query_text: str):
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}

    ----
    Answer the question based on the above context: {question}
    """

    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score, in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt =  prompt_template.format(context=context_text, question=query_text)
    return prompt

prompt = query_rag("how many cards has each palyer at the beginning of a game?")

print(prompt) #as you see you have the prompt template where the context gives you the k chunks with higher score 
#so more close to the query text

#now that you have the prompt you can work on the response
from langchain_community.llms.ollama import Ollama

model= Ollama(model='llama3') #choose a model you have on your machine
response_text = model.invoke(prompt)
print(response_text)