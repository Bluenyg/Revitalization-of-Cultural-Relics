import os

# --- 0. 基础配置与环境设置 ---
# 必须在导入任何 langchain/chromadb 库之前设置
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ALLOW_RESET"] = "True"

import streamlit as st
import json
import glob
import re
import shutil
from datetime import datetime

# LangChain & RAG 库
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

# --- 页面配置 ---
st.set_page_config(
    page_title="国宝AI活化工作台 (Pro Max版)",
    page_icon="🏺",
    layout="wide"
)

# 定义路径
CHROMA_DB_DIR = "./chroma_db"
HISTORY_DIR = "./chat_history"
KNOWLEDGE_BASE_DIR = "./knowledge_base"  # 新增：知识库源文件存储目录
OLLAMA_URL = "http://localhost:11434"

# 确保目录存在
for directory in [HISTORY_DIR, CHROMA_DB_DIR, KNOWLEDGE_BASE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- 1. 深度角色定义 ---
ROLE_DEFINITIONS = {
    "专家学者": {
        "description": "像一位学术泰斗。你的回答应当系统化、逻辑严密、论述充分。",
        "focus": "历史与考古、工艺与材料、文献考据。",
        "instruction": "直接写'介绍一下...'时，不要泛泛而谈，要像做专题报告一样。引用考古发掘成果，分析其在历史长河中的地位。"
    },
    "研究型助理": {
        "description": "高效的研究助手。你的任务是快速、准确地搜集、整理、加工信息。",
        "focus": "资料汇总、数据分析、条理清晰。",
        "instruction": "你可以自主选择分析角度。形成的汇报材料必须结构清晰（使用Markdown列表），帮助用户快速理解核心知识点。"
    },
    "交互设计师": {
        "description": "前瞻性的体验创造者。关注'如何让文物被感知'和'观众将如何体验它'。",
        "focus": "展览交互设计、沉浸体验、数字空间(AR/VR)、五感体验。",
        "instruction": "基于创意和技术，提出体验性的解决方案。思考如何通过手势、声音、触控等反人机交互手段，让文物'活'起来。"
    },
    "符号学者": {
        "description": "跨时空的创意顾问。深度破译文物符号的文化与哲学内涵。",
        "focus": "符号与隐喻、纹样解读、宗教与仪式、哲学与设计。",
        "instruction": "不要只看表面，要解读符号背后的隐喻（如龙代表皇权）。将古老的智慧转化为现代设计的新颖叙事。"
    },
    "用户体验研究员": {
        "description": "关注用户痛点与需求的研究者。系统性分析用户在互动前、中、后的所有触点。",
        "focus": "用户画像、痛点分析、触点(Touchpoints)、服务蓝图。",
        "instruction": "提供结构化的研究建议。思考用户为什么'看不懂'、'记不住'。关注如何通过设计解决这些认知障碍。"
    },
    "策展人": {
        "description": "故事的讲述者与传播者。思考如何让文物进入大众视野并成为'爆款'。",
        "focus": "叙事传播、品牌文化、社会热点结合、短视频传播。",
        "instruction": "思考如何将学术内容转化为大众可感知的'故事'。关注抖音等平台的传播规律（完播率、话题性）。"
    },
    "情感化设计研究员": {
        "description": "心理学与设计的结合者。关注'功能'如何转化为'情感连接'。",
        "focus": "情感共鸣、心理学、诗意表达、生命体验。",
        "instruction": "挖掘文物背后的情感价值。思考如何通过设计引发观众的'惊叹'、'感动'或'沉思'。"
    }
}

PROFESSIONAL_ANGLES = """
在分析时，请综合考虑以下维度：
1. [历史与考古]: 出土背景、断代依据、社会制度。
2. [工艺与材料]: 制作方法、材料来源、技术水平。
3. [色彩]: 颜色美学、等级象征、矿物颜料。
4. [器形]: 功能与审美的结合、礼制规范。
5. [符号与隐喻]: 纹样的文化寓意、宗教内涵。
6. [诗意表达]: 相关的诗词歌赋、文学意象。
7. [数字空间]: AR/VR、反人机交互的可能性。
"""


# --- 2. 核心功能函数 ---

def parse_and_render_message(text):
    """解析 <think> 标签并渲染"""
    pattern = r"<think>(.*?)</think>(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thought, answer = match.group(1).strip(), match.group(2).strip()
        if thought:
            with st.expander("💭 模型思考过程 (点击展开)", expanded=False):
                st.markdown(thought)
        if answer:
            st.markdown(answer)
        else:
            st.info("模型仅输出了思考过程。")
    else:
        clean_text = text.replace("<think>", "**[开始思考]**\n").replace("</think>", "\n**[思考结束]**\n")
        st.markdown(clean_text)


def get_vector_store():
    """获取向量数据库实例"""
    embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)


def process_uploaded_file(uploaded_file):
    """处理上传文件：保存到目录 -> 加载 -> 切分 -> 存入数据库"""
    try:
        # 1. 保存文件到 knowledge_base 目录 (持久化存储)
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. 选择加载器
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = TextLoader(file_path, autodetect_encoding=True)

        docs = loader.load()

        # 3. 切分文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        # 4. 存入数据库
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)
        vector_store.persist()

        return True, f"✅ 成功入库：{len(chunks)} 个知识块"
    except Exception as e:
        return False, str(e)


def delete_document(filename):
    """删除文档：从数据库移除向量 -> 从磁盘删除文件"""
    try:
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)

        # 1. 从 ChromaDB 中删除 (根据 source metadata)
        vector_store = get_vector_store()
        # Chroma 的 collection.delete 可以根据 where 条件删除
        vector_store._collection.delete(where={"source": file_path})
        vector_store.persist()

        # 2. 从磁盘删除文件
        if os.path.exists(file_path):
            os.remove(file_path)
            return True, f"🗑️ 已删除: {filename}"
        else:
            return True, f"⚠️ 文件已从库中移除，但磁盘上未找到原文件: {filename}"

    except Exception as e:
        return False, f"删除失败: {str(e)}"


def get_uploaded_files():
    """获取已上传的文件列表"""
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        return []
    return sorted(os.listdir(KNOWLEDGE_BASE_DIR))


# 历史记录管理函数 (保持不变)
def save_chat_history(chat_id, messages):
    with open(os.path.join(HISTORY_DIR, f"{chat_id}.json"), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)


def load_chat_history(chat_id):
    path = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else []


def get_chat_history_list():
    files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def get_chat_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"][:12] + "..." if len(msg["content"]) > 12 else msg["content"]
    return "新对话"


def init_new_chat():
    st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是国宝活化助手。请上传资料或直接提问。"}]


# --- 3. 初始化 Session ---
if "current_chat_id" not in st.session_state:
    init_new_chat()


# --- 4. 加载资源 ---
@st.cache_resource
def load_resources():
    embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
    llm = ChatOllama(base_url=OLLAMA_URL, model="qwen3:14b", temperature=0.5)
    return embeddings, llm


try:
    embeddings, llm = load_resources()
except Exception as e:
    st.error(f"模型连接失败: {e}")
    st.stop()

# --- 5. 侧边栏 UI ---
with st.sidebar:
    st.title("🏺 国宝AI活化")

    # === 模块 A: 知识库管理 ===
    with st.expander("📚 知识库管理 (上传/查看/删除)", expanded=False):
        # 1. 上传区域
        uploaded_file = st.file_uploader("上传新资料 (PDF/MD/TXT)", type=["pdf", "md", "txt"])
        if uploaded_file and st.button("开始学习", key="upload_btn"):
            with st.spinner("正在阅读并存入大脑..."):
                success, msg = process_uploaded_file(uploaded_file)
                if success:
                    st.success(msg)
                    st.cache_resource.clear()  # 清除缓存，确保下次检索能用新数据
                    st.rerun()  # 刷新页面显示新文件列表
                else:
                    st.error(f"学习失败: {msg}")

        st.divider()

        # 2. 文件列表与删除区域
        st.caption("📂 已收录文档列表")
        existing_files = get_uploaded_files()

        if not existing_files:
            st.info("暂无文档")
        else:
            for filename in existing_files:
                col_f1, col_f2 = st.columns([0.8, 0.2])
                with col_f1:
                    st.text(filename)
                with col_f2:
                    if st.button("❌", key=f"del_doc_{filename}", help="删除此文档"):
                        with st.spinner("正在删除..."):
                            success, msg = delete_document(filename)
                            if success:
                                st.success(msg)
                                st.cache_resource.clear()
                                st.rerun()
                            else:
                                st.error(msg)

    st.divider()

    # === 模块 B: 对话管理 ===
    if st.button("➕ 新建对话", use_container_width=True):
        init_new_chat()
        st.rerun()

    st.subheader("📜 历史记录")
    files = get_chat_history_list()
    if not files: st.caption("暂无记录")

    for file_path in files:
        file_name = os.path.basename(file_path)
        chat_id = file_name.replace(".json", "")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                msgs = json.load(f)
            title = get_chat_title(msgs)
            date_str = f"{chat_id[4:6]}/{chat_id[6:8]} {chat_id[9:11]}:{chat_id[11:13]}"
        except:
            title = "未知对话"
            date_str = ""

        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            prefix = "📂 " if st.session_state.current_chat_id == chat_id else ""
            if st.button(f"{prefix}{title}\n{date_str}", key=f"load_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = msgs
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{chat_id}"):
                os.remove(file_path)
                if st.session_state.current_chat_id == chat_id: init_new_chat()
                st.rerun()

    st.divider()

    # === 模块 C: 角色选择 ===
    selected_role = st.selectbox("🎭 选择分析角色", list(ROLE_DEFINITIONS.keys()))
    role_info = ROLE_DEFINITIONS[selected_role]
    st.info(f"**{selected_role}**\n\n{role_info['description']}")

# --- 6. 主界面与 RAG 逻辑 ---

st.header(f"当前会话: {get_chat_title(st.session_state.messages)}")

# 6.1 动态加载数据库
if os.path.exists(CHROMA_DB_DIR):
    # 每次重新加载 vector_store 以确保获取最新状态
    vector_store = get_vector_store()
    try:
        cnt = vector_store._collection.count()
        # 如果文档很少，就减少 k 值，避免报错
        k = min(4, cnt) if cnt > 0 else 0
    except:
        k = 0

    if k > 0:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
    else:
        retriever = None
else:
    retriever = None

# 6.2 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            parse_and_render_message(msg["content"])
        else:
            st.markdown(msg["content"])

# 6.3 构建 Prompt
role_config = ROLE_DEFINITIONS[selected_role]

system_template = f"""
你现在的身份是：**{selected_role}**。
{role_config['description']}

**你的核心关注点：**
{role_config['focus']}

**回复指导原则：**
{role_config['instruction']}

**专业分析维度参考：**
{PROFESSIONAL_ANGLES}

请根据以下【参考资料】来回答用户的【问题】。
如果资料中没有答案，请运用你的专业知识进行合理推演，但必须声明这是推演。

---
【参考资料】：
{{context}}
---
【用户问题】：
{{question}}
"""

prompt = ChatPromptTemplate.from_template(system_template)

# 6.4 处理输入
if user_input := st.chat_input("请输入关于文物的创意或问题..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_chat_history(st.session_state.current_chat_id, st.session_state.messages)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        if retriever:
            chain = (
                    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                     "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )
        else:
            chain = (
                    {"context": lambda x: "暂无本地知识库，请依靠通用知识回答。", "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )

        try:
            with st.spinner(f"{selected_role} 正在调动知识库进行分析..."):
                for chunk in chain.stream(user_input):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

                placeholder.empty()
                parse_and_render_message(full_response)
        except Exception as e:
            st.error(f"生成出错: {e}")
            full_response = "抱歉，系统出了点小差错。"

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat_history(st.session_state.current_chat_id, st.session_state.messages)

# CSS 优化
st.markdown("""
<style>
    .stButton>button {border-radius: 8px;}
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] button {
        text-align: left; border: 1px solid #eee;
    }
    .streamlit-expanderHeader {
        background-color: #f8f9fa; border-radius: 5px; font-size: 0.9em;
    }
    /* 调整删除按钮样式 */
    div[data-testid="column"]:nth-of-type(2) button {
        color: #ff4b4b;
        border-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)