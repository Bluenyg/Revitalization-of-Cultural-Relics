import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 定义知识库目录和 ChromaDB 持久化目录
KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHROMA_DB_DIR = "./chroma_db"

# 1. 定义 Embedding 模型 (使用 Ollama)
# 确保 nomic-embed-text 已经 ollama pull
print("初始化 Embedding 模型...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. 加载文档
# 我们使用 DirectoryLoader 来加载 'knowledge_base' 目录下的所有文档
# 我们为 PDF 单独指定加载器
print(f"正在从 '{KNOWLEDGE_BASE_DIR}' 加载文档...")
loader = DirectoryLoader(
    KNOWLEDGE_BASE_DIR,
    glob="**/*.*", # 加载所有类型的文件
    loader_cls=lambda path: PyPDFLoader(path) if path.endswith('.pdf') else None,
    use_multithreading=True,
    show_progress=True
)
documents = loader.load()

if not documents:
    print("未找到任何文档。请检查 'knowledge_base' 目录。")
    exit()

print(f"成功加载 {len(documents)} 篇文档。")

# 3. 切分文档 (Chunking)
print("正在切分文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个块的大小
    chunk_overlap=200   # 块之间的重叠
)
chunks = text_splitter.split_documents(documents)
print(f"文档被切分为 {len(chunks)} 个块。")

# 4. 向量化并存入 ChromaDB
# ChromaDB 将在服务器磁盘上的 'chroma_db' 目录中创建并持久化
print("正在将数据存入 ChromaDB (这可能需要一些时间)...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB_DIR
)

print("="*40)
print(f"数据入库完成！向量数据库已保存在: {CHROMA_DB_DIR}")
print("="*40)