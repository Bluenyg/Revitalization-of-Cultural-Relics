import sys
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 定义 ChromaDB 目录 (必须和 ingest.py 一致)
CHROMA_DB_DIR = "./chroma_db"

# 1. 定义 LLM (qwen3) 和 Embedding (nomic)
print("初始化 LLM (qwen3:14b) 和 Embedding (nomic-embed-text)...")
llm = ChatOllama(model="qwen3:14b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. 加载持久化的 ChromaDB
print(f"正在从 '{CHROMA_DB_DIR}' 加载向量数据库...")
try:
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
except Exception as e:
    print(f"加载数据库失败: {e}")
    print(f"请确保你已经先运行了 'ingest.py' 并且 '{CHROMA_DB_DIR}' 目录存在。")
    sys.exit(1)

print("数据库加载成功。")

# 3. 创建一个检索器 (Retriever)
# k=3 意味着每次检索3个最相关的文本块
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4. 定义 Prompt 模板
# 这是 RAG 的核心：我们指示 LLM 必须基于我们提供的 "Context" 来回答
RAG_PROMPT_TEMPLATE = """
**使用说明:**
你是一个专业的助手。请根据下面提供的 **[上下文信息]** 来回答用户的 **[问题]**。

**规则:**
1.  **严格基于上下文:** 你的回答必须完全基于提供的 [上下文信息]，不得依赖任何外部知识。
2.  **直接回答:** 如果 [上下文信息] 足够回答 [问题]，请直接回答。
3.  **无法回答:** 如果 [上下文信息] 没有包含回答 [问题] 所需的信息，请明确告知："根据我所掌握的资料，我无法回答这个问题。"

---
**[上下文信息]:**
{context}
---
**[问题]:**
{question}

**你的回答:**
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE
)

# 5. 创建 RAG 链 (RAG Chain)
# LangChain 的 LCEL 链式语法

def format_docs(docs):
    # 将检索到的文档块格式化为字符串
    return "\n\n".join(f"--- [相关资料 {i+1}] ---\n{doc.page_content}" for i, doc in enumerate(docs))

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. (可选) 创建一个用于 "追问" 的链，以防 RAG 检索不到内容
# (这部分较复杂，我们先用上面的基础 RAG 链)

def get_query_from_cli():
    # 从命令行获取问题
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    else:
        print("用法: python3 query.py '你的问题'")
        sys.exit(1)

# --- 主程序 ---
if __name__ == "__main__":
    question = get_query_from_cli()

    print(f"\n[用户问题]: {question}\n")
    print("[RAG 系统] 正在思考 (检索并生成答案)...")

    # 调用 RAG 链
    answer = rag_chain.invoke(question)

    print("\n[RAG 系统回答]:")
    print(answer)