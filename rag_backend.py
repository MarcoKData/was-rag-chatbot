import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock
import boto3
from typing import Any


def get_bedrock_runtime() -> Any:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

    return bedrock_runtime


def get_index() -> Any:
    data_load = PyPDFLoader("https://arxiv.org/pdf/1706.03762.pdf")

    data_split = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=100,
        chunk_overlap=10
    )

    bedrock_runtime = get_bedrock_runtime()

    data_embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v1"
    )

    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS
    )

    db_index = data_index.from_loaders([data_load])

    return db_index


def get_llm() -> Bedrock:
    bedrock_runtime = get_bedrock_runtime()

    llm = Bedrock(
        client=bedrock_runtime,
        model_id="meta.llama2-70b-chat-v1",
        model_kwargs={
            "temperature": 0.5,
            "top_p": 0.9,
            "max_gen_len": 512
        }
    )

    return llm


def rag_response(index, question: str, llm: Bedrock) -> Any:
    rag_query = index.query(
        question=question,
        llm=llm
    )

    return rag_query


"""print("Creating Index...")
idx = get_index()
print("Getting LLM...")
llm = get_llm()
print("Generating Result...")
result = rag_response(index=idx, question="What is attention?", llm=llm)
print(result)"""
