# RAG_uno_rules

#This code will use ollama3 to answer question about the card game UNO and its rules.

#First we read the pdf documents of the rules downloaded and save their text (we say that we "load data")

#now each pdf is saved inside document, a list that contains for each index the contect of the page of the pdf and its metadata(like page number and pdf name)

#it is time to split the text in chunks and save them into a databe in their embedded version (multidimensional coordinates)

#the function split_documents takes the loaded list as argunement, so it contains the text of the pages of the pdf an its realtive metadata,
#the chunk contains the splitted part of the page_content and its metadata, it is still a list

#now is time to define an embedding function for the chunks, remember to use the same embedding to store and to retrive the data otherwise it wont work

#locally I had trouble woth the localhost standard port
i have switched the ollama port to localhost to 8888 and pulled the model for embedding nomic-embed-text
OLLAMA_HOST=0.0.0.0:8888 ollama pull nomic-embed-text

#now we create the vector database with CHROMADB (openAI)

#now we define the prompt schema that will give the right context to the llm before answering

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}

    ----
    Answer the question based on the above context: {question}
    """



#as you see you have the prompt template where the context gives you the k chunks with higher score 
#so more close to the query text

#now that you have the prompt you can work on the response
