import os
import glob
import json
import time
import logging
from typing import List, Optional

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# =========================================================
# 1. Configuration
# =========================================================

MAX_RETRIES = 5
TEMPERATURE = 0.3

LLM_MODEL = "gpt-4o-2024-08-06"
EMBEDDING_MODEL = "text-embedding-3-small"

# basic / advance
MODE = "basic"

# Please set your benchmark file path manually
BENCHMARK_PATH = "./benchmarks/your_benchmark.xlsx"
BENCHMARK_SHEET = "Sheet1"

OUTPUT_FILENAME = f"results_{MODE}_{LLM_MODEL}.csv"

# Leave blank for GitHub public release
OPENAI_API_BASE = ""
OPENAI_API_KEY = ""

# Knowledge source paths
GUIDELINE_PDF_GLOB = "./Knowledge_Sources_Guideline/*.pdf"
ADVANCED_JSONL_GLOB = "./Knowledge_Sources_Advanced/*.jsonl"

# Vector DB paths
BASIC_PERSIST_DIRECTORY = "./chroma_db_basic"
ADVANCED_PERSIST_DIRECTORY = "./chroma_db_advanced"
HYBRID_PERSIST_DIRECTORY = "./chroma_db_hybrid"

# Optional conversational mode for RAG
ENABLE_CONVERSATIONAL_RAG = False


# =========================================================
# 2. API / Model Setup
# =========================================================

os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

llm = ChatOpenAI(
    model_name=LLM_MODEL,
    temperature=TEMPERATURE,
    max_tokens=4096
)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


# =========================================================
# 3. Prompts
# =========================================================

PROMPT_VANILLA = """
You are a biomedical expert. Answer the following question based on your internal knowledge.

Question: {question}
"""

PROMPT_COT = """
You are a biomedical expert. Answer the following question. First, provide your step-by-step reasoning process. Then, provide the final answer.

Let's think step by step.

Question: {question}

Reasoning:
[Your step-by-step reasoning here]

Final Answer:
[Your answer here]
"""

PROMPT_ROT = """
You are a biomedical expert. Answer the following question.

Question: {question}

Imagine 3 medical experts are solving this task. Each expert independently provides their step-by-step reasoning and final answer.
After all experts have finished, they discuss together, review and backtrack their previous reasoning steps, and finally reach a consensus on the final answer.

Please present:
[Expert 1's reasoning and answer],
[Expert 2's reasoning and answer],
[Expert 3's reasoning and answer],
[The discussion and the agreed final answer]
"""

RAG_TEMPLATE = """
You are a biomedical expert. Answer the following question based on the provided clinical guideline context and your internal knowledge.

Question: {question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)


# =========================================================
# 4. Document Loading
# =========================================================

def load_pdf_documents() -> List[Document]:
    pdf_paths = glob.glob(GUIDELINE_PDF_GLOB)
    all_documents = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        for doc in documents:
            meta = doc.metadata or {}
            meta.update({
                "source": pdf_path,
                "source_type": "pdf",
                "source_name": os.path.basename(pdf_path),
            })
            doc.metadata = meta

        all_documents.extend(documents)

    print(f"[PDF] Loaded {len(all_documents)} raw document pages")
    return all_documents


def load_jsonl_documents() -> List[Document]:
    jsonl_paths = glob.glob(ADVANCED_JSONL_GLOB)
    all_documents = []

    for jsonl_path in jsonl_paths:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {jsonl_path}"):
                data = json.loads(line)
                content = json.dumps(data, ensure_ascii=False)
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": jsonl_path,
                        "source_type": "jsonl",
                        "source_name": os.path.basename(jsonl_path),
                    }
                )
                all_documents.append(doc)

    print(f"[JSONL] Loaded {len(all_documents)} records")
    return all_documents


def split_pdf_documents(pdf_docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    pdf_chunks = text_splitter.split_documents(pdf_docs)
    print(f"[PDF] Split into {len(pdf_chunks)} chunks")
    return pdf_chunks


# =========================================================
# 5. Vector DB Creation / Loading
# =========================================================

def create_vector_db(documents: List[Document], persist_directory: str) -> Chroma:
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"[VectorDB] Created and persisted at: {persist_directory}")
    return vectordb


def load_vector_db(persist_directory: str) -> Chroma:
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectordb


def create_basic_vector_db() -> Chroma:
    pdf_docs = load_pdf_documents()
    pdf_chunks = split_pdf_documents(pdf_docs)
    return create_vector_db(pdf_chunks, BASIC_PERSIST_DIRECTORY)


def create_advanced_vector_db() -> Chroma:
    jsonl_docs = load_jsonl_documents()
    return create_vector_db(jsonl_docs, ADVANCED_PERSIST_DIRECTORY)


def create_hybrid_vector_db() -> Chroma:
    pdf_docs = load_pdf_documents()
    pdf_chunks = split_pdf_documents(pdf_docs)
    jsonl_docs = load_jsonl_documents()
    all_docs = pdf_chunks + jsonl_docs
    print(f"[Hybrid] Total chunks/documents: {len(all_docs)}")
    return create_vector_db(all_docs, HYBRID_PERSIST_DIRECTORY)


def check_and_prepare_db(persist_directory: str, db_type: str) -> Chroma:
    if not os.path.exists(persist_directory):
        print(f"[{db_type}] Vector DB not found. Creating...")
        if db_type == "basic":
            vectordb = create_basic_vector_db()
        elif db_type == "advanced":
            vectordb = create_advanced_vector_db()
        elif db_type == "hybrid":
            vectordb = create_hybrid_vector_db()
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")
    else:
        print(f"[{db_type}] Found existing Vector DB. Loading...")
        vectordb = load_vector_db(persist_directory)

    try:
        metadatas = vectordb.get()["metadatas"]
        sources = [m.get("source") for m in metadatas if m and "source" in m]
        print(f"[{db_type}] Sources used:", set(sources))
    except Exception as e:
        print(f"[{db_type}] Warning: unable to inspect metadata. {e}")

    return vectordb


# =========================================================
# 6. Build Retrievers / Chains
# =========================================================

def build_rag_chain(vectordb: Chroma):
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


def build_conversational_rag_chain(vectordb: Chroma):
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        return_source_documents=True,
        verbose=False
    )
    return conversational_chain


basic_vectordb = check_and_prepare_db(BASIC_PERSIST_DIRECTORY, "basic")
advanced_vectordb = check_and_prepare_db(ADVANCED_PERSIST_DIRECTORY, "advanced")
hybrid_vectordb = check_and_prepare_db(HYBRID_PERSIST_DIRECTORY, "hybrid")

basic_rag_chain, basic_retriever = build_rag_chain(basic_vectordb)
advanced_rag_chain, advanced_retriever = build_rag_chain(advanced_vectordb)
hybrid_rag_chain, hybrid_retriever = build_rag_chain(hybrid_vectordb)

basic_conversational_chain = None
advanced_conversational_chain = None
hybrid_conversational_chain = None

if ENABLE_CONVERSATIONAL_RAG:
    basic_conversational_chain = build_conversational_rag_chain(basic_vectordb)
    advanced_conversational_chain = build_conversational_rag_chain(advanced_vectordb)
    hybrid_conversational_chain = build_conversational_rag_chain(hybrid_vectordb)


# =========================================================
# 7. Inference Functions
# =========================================================

def fetch_prompts_response(question: str, max_retries: int, prompt_template: str, sleep_duration: int = 1) -> Optional[str]:
    retries = 0
    content = prompt_template.format(question=question)

    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": content}],
                temperature=TEMPERATURE,
                model=LLM_MODEL,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error during API call: {str(e)}")
            retries += 1
            time.sleep(sleep_duration)
            logging.info(f"Retry {retries}/{max_retries}...")
            if retries == max_retries:
                raise Exception("Max retries reached, failing...")

    return None


def fetch_rag_response(question: str, rag_chain, max_retries: int, sleep_duration: int = 1) -> Optional[str]:
    retries = 0

    while retries < max_retries:
        try:
            answer = rag_chain.invoke(question)
            return answer
        except Exception as e:
            logging.error(f"Error during RAG call: {str(e)}")
            retries += 1
            time.sleep(sleep_duration)
            logging.info(f"Retry {retries}/{max_retries}...")
            if retries == max_retries:
                raise Exception("Max retries reached for RAG, failing...")

    return None


def fetch_conversational_rag_response(question: str, conversational_chain, max_retries: int, sleep_duration: int = 1) -> Optional[str]:
    retries = 0

    while retries < max_retries:
        try:
            result = conversational_chain.invoke({"question": question})
            return result["answer"]
        except Exception as e:
            logging.error(f"Error during conversational RAG call: {str(e)}")
            retries += 1
            time.sleep(sleep_duration)
            logging.info(f"Retry {retries}/{max_retries}...")
            if retries == max_retries:
                raise Exception("Max retries reached for conversational RAG, failing...")

    return None


def fetch_vanilla_response(question: str) -> Optional[str]:
    return fetch_prompts_response(question, MAX_RETRIES, PROMPT_VANILLA)


def fetch_cot_response(question: str) -> Optional[str]:
    return fetch_prompts_response(question, MAX_RETRIES, PROMPT_COT)


def fetch_rot_response(question: str) -> Optional[str]:
    return fetch_prompts_response(question, MAX_RETRIES, PROMPT_ROT)


def fetch_basic_rag_response(question: str) -> Optional[str]:
    if ENABLE_CONVERSATIONAL_RAG and basic_conversational_chain is not None:
        return fetch_conversational_rag_response(question, basic_conversational_chain, MAX_RETRIES)
    return fetch_rag_response(question, basic_rag_chain, MAX_RETRIES)


def fetch_advanced_rag_response(question: str) -> Optional[str]:
    if ENABLE_CONVERSATIONAL_RAG and advanced_conversational_chain is not None:
        return fetch_conversational_rag_response(question, advanced_conversational_chain, MAX_RETRIES)
    return fetch_rag_response(question, advanced_rag_chain, MAX_RETRIES)


def fetch_hybrid_rag_response(question: str) -> Optional[str]:
    if ENABLE_CONVERSATIONAL_RAG and hybrid_conversational_chain is not None:
        return fetch_conversational_rag_response(question, hybrid_conversational_chain, MAX_RETRIES)
    return fetch_rag_response(question, hybrid_rag_chain, MAX_RETRIES)


# =========================================================
# 8. Benchmark Parsing
# =========================================================

def parse_basic_row(row):
    question_type = row["Type"]
    domain_topic = row["Domain Topic"]

    if question_type == "True or false question":
        question = question_type + ": " + row["Question"]
    else:
        question = row["Question"]

    ground_truth = row["Reference Answer"]

    return {
        "Type": question_type,
        "Question": question,
        "Domain Topic": domain_topic,
        "Reference Answer": ground_truth
    }


def parse_advance_row(row):
    task = row["Task"]
    subtask = row["Subtask"]
    question = row["Question"]
    ground_truth = row["Reference Answer"]
    reference_article = row["Reference Article"]

    return {
        "Task": task,
        "Subtask": subtask,
        "Question": question,
        "Reference Answer": ground_truth,
        "Reference Article": reference_article
    }


# =========================================================
# 9. Main Benchmark Loop
# =========================================================

def main():
    if not BENCHMARK_PATH:
        raise ValueError("BENCHMARK_PATH is empty. Please set your benchmark file path.")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is empty. Please fill it before running.")

    benchmark = pd.read_excel(BENCHMARK_PATH, sheet_name=BENCHMARK_SHEET)
    results = []

    for _, row in tqdm(benchmark.iterrows(), total=benchmark.shape[0]):
        if MODE == "basic":
            parsed = parse_basic_row(row)
        elif MODE == "advance":
            parsed = parse_advance_row(row)
        else:
            raise ValueError("MODE must be either 'basic' or 'advance'")

        question = parsed["Question"]

        vanilla_response = fetch_vanilla_response(question)
        cot_response = fetch_cot_response(question)
        rot_response = fetch_rot_response(question)
        basic_rag_response = fetch_basic_rag_response(question)
        advanced_rag_response = fetch_advanced_rag_response(question)
        hybrid_rag_response = fetch_hybrid_rag_response(question)

        result_row = {
            **parsed,
            "Vanilla_response": vanilla_response,
            "COT_response": cot_response,
            "ROT_response": rot_response,
            "Basic_RAG_response": basic_rag_response,
            "Advanced_RAG_response": advanced_rag_response,
            "Hybrid_RAG_response": hybrid_rag_response,
        }

        results.append(result_row)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILENAME, index=False, encoding="utf-8-sig")
    print(f"Saved results to: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
