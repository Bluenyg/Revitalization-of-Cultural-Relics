import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 配置路径
KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHROMA_DB_DIR = "./chroma_db"


def create_vector_db():
    # 1. 检查目录
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        print(f"请将'舞马衔杯银壶'的 .md 资料放入 {KNOWLEDGE_BASE_DIR} 目录中！")
        return

    # 2. 初始化 Embedding 模型 (连接服务器本地的 Ollama)
    print("正在初始化 Embedding 模型 (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(
        base_url="http://localhost:11434",  # 服务器本地地址
        model="nomic-embed-text"
    )

    # 3. 加载文档 (强制使用 UTF-8 读取 Markdown)
    print("正在加载文档...")
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True}
    )
    docs = loader.load()

    if not docs:
        print("未找到文档，请检查目录！")
        return

    # 4. 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"共切分出 {len(chunks)} 个知识块。")

    # 5. 存入向量数据库
    print("正在写入 ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print(f"✅ 知识库构建完成！保存路径: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    create_vector_db()