# =================================================================
# 1. 导入所有需要的“工具包”
# =================================================================
import os
import glob
import time
import logging
import json # 【已添加】导入json库
import re   # 【已添加】导入re库
import requests # 【已添加】导入requests库
import pandas as pd
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings # 两个都导入以防万一
from langchain.vectorstores import Chroma

# =================================================================
# 2. 全局配置 (Global Configuration)
# =================================================================
# --- LLM 和 Embedding 模型配置 ---
LLM_MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "text-embedding-3-small" # Embedding模型名称
MAX_RETRIES = 5
TEMPERATURE = 0.3

# --- API 关键信息 ---
# 【已修改】将API信息放在这里，更清晰
API_URL = ""
API_KEY = ""

# --- 文件和路径配置 ---
PERSIST_DIRECTORY = r"chroma_db"
BENCHMARK_FILE = r"MSIC_basic_bench.xlsx"
OUTPUT_FOLDER = r""
OUTPUT_FILENAME = os.path.join(OUTPUT_FOLDER, f"MSIC_basic_bench_results-{LLM_MODEL}.csv")


# 【已修改】设置Langchain需要的环境变量，即使我们不直接用它的LLM调用，Embedding可能需要

os.environ["OPENAI_API_KEY"] = API_KEY
EMBEDDING_API_BASE = ""
# --- Prompt 模板定义 ---
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
You are a biomedical expert. Answer the following question based ONLY on the provided clinical guideline context.

Question: {question}

Context:
---
{context}
---
"""

# =================================================================
# 3. 向量数据库 (Vector Database) 创建与加载
# =================================================================
def create_and_save_vector_db():
    print("--- 步骤 1/3: 未发现本地向量数据库，开始创建... ---")
    pdf_paths = glob.glob('Knowledge_Sources/*.pdf')
    print(f"  > 发现 {len(pdf_paths)} 个PDF文件: {[os.path.basename(p) for p in pdf_paths]}")
    
    all_documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
    print(f'  > PDF加载完成，共切分成 {len(all_documents)} 个文档页面。')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    print(f'  > 文档块切分完成，共得到 {len(texts)} 个文本块。')

    print("  > 开始创建并保存向量数据库，这可能需要一些时间...")
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_base=EMBEDDING_API_BASE), # 指定模型
        persist_directory=PERSIST_DIRECTORY
    )
    vectordb.persist()
    print("--- 向量数据库创建并保存成功！ ---")
    return vectordb

def load_vector_db():
    print("--- 步骤 1/3: 发现已保存的向量数据库，直接加载... ---")
    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_base=EMBEDDING_API_BASE) # 指定模型
    )
    metadatas = vectordb.get().get('metadatas', [])
    if metadatas:
        sources = set(m.get('source') for m in metadatas if 'source' in m)
        print("  > 用于检索的文件:", [os.path.basename(s) for s in sources])
    else:
        print("  > 向量数据库为空或无法获取元数据。")
    print("--- 向量数据库加载成功！ ---")
    return vectordb

if not os.path.exists(PERSIST_DIRECTORY):
    vectordb = create_and_save_vector_db()
else:
    vectordb = load_vector_db()

# =================================================================
# 4. API 调用函数 (统一使用 requests)
# =================================================================

# 【已修改】这是你验证过成功的API调用函数，我们用它来处理所有非RAG的请求
def fetch_prompts_response(question, PROMPT, max_retries=MAX_RETRIES, sleep_duration=2):
    content = PROMPT.format(question=question)
    headers = {
       'Authorization': f'Bearer {API_KEY}',
       'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4096,
        "temperature": TEMPERATURE,
        "stream": False
    })

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=payload, timeout=180)
            response.raise_for_status()
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content'].strip()
            else:
                raise ValueError("API返回格式不正确")
        except Exception as e:
            logging.error(f"API 调用失败 (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(sleep_duration)
            else:
                return f"ERROR: API call failed after {max_retries} retries." # 返回错误信息而不是崩溃
    return None

# 【已修改】这是新的RAG响应函数，它手动执行检索、构建Prompt、然后调用上面的函数
def fetch_RAG_response(question, max_retries=MAX_RETRIES):
    try:
        # 步骤 1: 使用Langchain的retriever进行文档检索
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        retrieved_docs = retriever.get_relevant_documents(question)
        
        # 步骤 2: 将检索到的文档内容格式化为上下文
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 步骤 3: 使用我们自己的API函数来获取回答
        # 注意这里我们传入的是RAG_TEMPLATE，而不是其他模板
        answer = fetch_prompts_response(
            question=question,
            PROMPT=RAG_TEMPLATE.format(question="{question}", context=context), # 手动填充context
            max_retries=max_retries
        )
        return answer
    except Exception as e:
        logging.error(f"RAG流程失败: {e}")
        return f"ERROR: RAG process failed: {e}"

# =================================================================
# 5. 主执行逻辑 (Main Execution Logic)
# =================================================================
# 定义错误信息，方便后面判断
ERROR_MSG = "ERROR: API call failed after 5 retries."

# 加载完整的 benchmark 表格
basic_benchmark = pd.read_excel(BENCHMARK_FILE, sheet_name="Sheet1")

# 初始化结果列表和已完成问题集合
results = []
completed_questions = set()

# 【断点续跑逻辑】
if os.path.exists(OUTPUT_FILENAME):
    print(f"--- 发现已存在的结果文件 '{OUTPUT_FILENAME}'，启动断点续跑模式... ---")
    df_existing = pd.read_csv(OUTPUT_FILENAME)
    
    # 找出所有4个响应列都不包含错误信息的行，这些是完全成功的
    successful_rows = df_existing[
        (~df_existing['Vanilla_response'].astype(str).str.contains("ERROR", na=False)) &
        (~df_existing['COT_response'].astype(str).str.contains("ERROR", na=False)) &
        (~df_existing['ROT_response'].astype(str).str.contains("ERROR", na=False)) &
        (~df_existing['RAG_response'].astype(str).str.contains("ERROR", na=False))
    ]
    
    # 将已完成的问题加入集合，方便快速查找
    completed_questions = set(successful_rows['question'])
    print(f"  > 已成功处理 {len(completed_questions)} / {len(basic_benchmark)} 个问题，将跳过它们。")
    
    # 将已成功的结果先加载到结果列表中
    results = successful_rows.to_dict('records')
else:
    print(f"--- 未发现结果文件，将从头开始运行... ---")

# 开始处理...
print(f"\n--- 开始使用模型 '{LLM_MODEL}' 处理 Benchmark... ---")
for index, row in tqdm(basic_benchmark.iterrows(), total=basic_benchmark.shape[0], desc="处理Benchmark问题"):
    question = str(row['question'])

    # 【断点续跑核心检查】
    if question in completed_questions:
        continue # 如果这个问题已经处理过，直接跳到下一个

    # --- 如果是未完成的问题，则正常执行所有API调用 ---
    question_type = row['type']
    full_question = f"{question_type}: {question}"
    domain_topic = row['Domain_Topic']
    ground_truth = row['answer']

    print(f"\n继续处理问题 {index + 1}: {full_question[:100]}...")

    vanilla_response = fetch_prompts_response(full_question, PROMPT_VANILLA)
    cot_response = fetch_prompts_response(full_question, PROMPT_COT)
    rot_response = fetch_prompts_response(full_question, PROMPT_ROT)
    rag_response = fetch_RAG_response(full_question)

    results.append({
        "type": question_type,
        "question": question,
        "domain_topic": domain_topic,
        "ground_truth": ground_truth,
        "Vanilla_response": vanilla_response,
        "COT_response": cot_response,
        "ROT_response": rot_response,
        "RAG_response": rag_response,
    })

# =================================================================
# 6. 保存结果
# =================================================================
print(f"\n--- 所有问题处理完毕，正在合并并保存最终结果... ---")
final_df = pd.DataFrame(results)

# 【重要】为了保证顺序和原始benchmark一致，我们做一个排序
# 创建一个问题到顺序的映射
question_order = {str(q): i for i, q in enumerate(basic_benchmark['question'])}
# 根据这个顺序给 final_df 排序
final_df['sort_order'] = final_df['question'].map(question_order)
final_df = final_df.sort_values('sort_order').drop(columns=['sort_order'])

final_df.to_csv(OUTPUT_FILENAME, index=False, encoding="utf-8-sig")
print(f"🎉 任务完成！结果已更新并保存至: {OUTPUT_FILENAME}")