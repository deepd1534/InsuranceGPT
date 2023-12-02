import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch
import base64
import textwrap

checkpoint = "/home/imdeep/LocalLLM/LaMini-T5-223M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.float32,
)

@st.cache_resource
def llm_pipeline():
    """ Defining pipeline for Text Generation"""
    
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer=tokenizer,
        max_length= 256,
        do_sample = True,
        temperature= 0.5,
        top_p = 0.95
    )
    
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# @st.cache_resource
# def qa_llm():
#     llm = llm_pipeline()
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = Chroma(persist_directory="db", embedding_function=embeddings)
#     retriever = db.as_retriever(search_kwargs={"k": 3})
    
#     qa = RetrievalQA.from_chain_type(
#         llm = llm,
#         chain_type = "stuff",
#         retriever = retriever,
#         return_source_documentes = True
#     )
 
@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  # Fix the typo here
    )
    return qa
 
    
def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text


def main():
    st.title('Motor Insurance ChatBot 🚗💥💸💼📝')
    with st.expander("About the App"):
        st.markdown(
    """
    This is a chatbot for insurance
    """
        )
        
    question = st.text_area("Enter your Query:")
    if st.button("Search"):
        st.info("Your Question: " + question)
        st.info("Your Answer: ")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)
    
    
if __name__ == "__main__":
    main()
    
    
    

