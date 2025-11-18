import os

# --- 0. 基础配置与环境设置 ---
# 必须在导入任何 langchain/chromadb 库之前设置，防止 Telemetry 报错
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ALLOW_RESET"] = "True"

import streamlit as st
import json
import glob
import re  # 引入正则库用于解析 <think> 标签
from datetime import datetime

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 页面配置 ---
st.set_page_config(
    page_title="国宝AI活化工作台",
    page_icon="🐴",
    layout="wide"
)

# 定义路径
CHROMA_DB_DIR = "./chroma_db"
HISTORY_DIR = "./chat_history"
OLLAMA_URL = "http://localhost:11434"

# 确保历史记录目录存在
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)


# --- 1. 辅助函数：解析与渲染消息 ---

def parse_and_render_message(text):
    """
    核心渲染函数：
    检测文本中是否包含 <think> 标签。
    如果有，将思考过程放入 st.expander 折叠框，
    将正式回答放入 markdown。
    """
    # 使用非贪婪匹配提取 <think> 内容
    # re.DOTALL 让 . 也能匹配换行符
    pattern = r"<think>(.*?)</think>(.*)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        thought_content = match.group(1).strip()
        answer_content = match.group(2).strip()

        # 1. 渲染思考过程（默认折叠，expanded=False）
        if thought_content:
            with st.expander("💭 模型思考过程 (点击展开)", expanded=False):
                st.markdown(thought_content)

        # 2. 渲染正式回答
        if answer_content:
            st.markdown(answer_content)
        else:
            # 只有思考没有回答的情况（极少见）
            st.info("模型仅输出了思考过程，未生成最终回答。")

    else:
        # 如果没有标签，直接显示全文
        # 处理流式输出中可能出现的未闭合标签（仅作简单处理，避免显示乱码）
        clean_text = text.replace("<think>", "**[开始思考]**\n").replace("</think>", "\n**[思考结束]**\n")
        st.markdown(clean_text)


# --- 2. 历史记录管理 ---

def save_chat_history(chat_id, messages):
    """将当前对话保存到 JSON 文件"""
    file_path = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)


def load_chat_history(chat_id):
    """读取历史对话"""
    file_path = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def get_chat_history_list():
    """获取所有历史记录文件，按修改时间排序"""
    files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def get_chat_title(messages):
    """根据第一条用户消息生成标题"""
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"][:15] + "..."
    return "新对话"


# --- 3. 初始化 Session State ---

if "current_chat_id" not in st.session_state:
    new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_chat_id = new_id
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是你的国宝活化助理。关于“舞马衔杯仿皮囊式银壶”，你有什么大胆的创作想法？"}
    ]


# --- 4. 加载 RAG 模型 (缓存) ---

@st.cache_resource
def load_rag_chain():
    # Embedding
    embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")

    # Vector DB
    if not os.path.exists(CHROMA_DB_DIR):
        st.error("❌ 未找到向量数据库。请先在终端运行 `python3 ingest.py` 生成知识库！")
        st.stop()

    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

    # 智能调整 k 值
    try:
        collection_count = vector_store._collection.count()
        k_val = min(4, collection_count)
        if k_val == 0: k_val = 1
    except:
        k_val = 4

    retriever = vector_store.as_retriever(search_kwargs={"k": k_val})

    # LLM
    llm = ChatOllama(base_url=OLLAMA_URL, model="qwen3:14b", temperature=0.3)
    return retriever, llm


try:
    retriever, llm = load_rag_chain()
except Exception as e:
    st.error(f"无法连接模型，请检查 Ollama 服务。错误: {e}")
    st.stop()

# --- 5. 侧边栏 UI ---

with st.sidebar:
    st.image("https://img.icons8.com/color/96/museum.png", width=80)
    st.title("🐴 国宝画重点")

    st.subheader("🗂️ 对话管理")

    if st.button("➕ 新建对话", use_container_width=True):
        new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.current_chat_id = new_id
        st.session_state.messages = [
            {"role": "assistant", "content": "已开启新对话。请告诉我你的新想法！"}
        ]
        st.rerun()

    with st.expander("📜 历史记录 (点击切换)", expanded=True):
        files = get_chat_history_list()
        for file_path in files:
            file_name = os.path.basename(file_path)
            chat_id = file_name.replace(".json", "")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    msgs = json.load(f)
                title = get_chat_title(msgs)
                display_label = f"{title}\n({chat_id[4:8]}-{chat_id[9:11]})"
            except:
                display_label = chat_id

            if st.button(display_label, key=chat_id, use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = msgs
                st.rerun()

    st.divider()

    role = st.selectbox(
        "🎭 选择 AI 助理角色",
        ("专家学者 (严谨考据)", "交互设计师 (体验创新)", "符号学者 (文化隐喻)", "策展人 (叙事传播)")
    )

    role_definitions = {
        "专家学者 (严谨考据)": "你是一位严谨的历史学家。你的重点是考证史实。如果用户的创意不符合唐代历史或文物事实，请指出并提供依据。",
        "交互设计师 (体验创新)": "你是一位前卫的交互设计师。请评估用户的创意是否具有互动性（如重力感应、手势），并给出优化建议。",
        "符号学者 (文化隐喻)": "你是一位符号学专家。请解读文物背后的文化隐喻（如'胡汉融合'），帮助用户深化作品内涵。",
        "策展人 (叙事传播)": "你是一位新媒体策展人。请从抖音传播的角度（完播率、话题性）评估用户的方案。"
    }
    st.info(f"**当前设定：**\n{role_definitions[role]}")

# --- 6. 主聊天界面 ---

st.header(f"当前会话: {get_chat_title(st.session_state.messages)}")

# 6.1 显示历史消息（使用新的渲染函数）
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            parse_and_render_message(message["content"])
        else:
            st.markdown(message["content"])

# RAG Prompt
template = f"""
你现在的身份是：**{role}**。
请根据以下【上下文信息】（关于舞马衔杯银壶的史实）来分析用户的【创意方案】。

**任务要求：**
1.  **纠偏：** 如果用户的描述与史实冲突，请务必温和地指出并纠正。
2.  **深化：** 基于你的角色（{role}），为用户的方案提供专业的优化建议。

---
【上下文信息】：
{{context}}
---
【用户的创意方案】：
{{question}}
---
**你的分析与建议：**
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# 6.2 处理新消息输入
if user_input := st.chat_input("请输入你的创意方案..."):
    # 用户消息上屏
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_chat_history(st.session_state.current_chat_id, st.session_state.messages)

    # 助手消息生成
    with st.chat_message("assistant"):
        # 创建一个空的容器，用于实时更新
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner(f"{role} 正在思考与查阅资料..."):
            try:
                # 流式输出
                for chunk in rag_chain.stream(user_input):
                    full_response += chunk
                    # 流式过程中，暂时显示原始内容（包含 <think> 标签）
                    # 这样可以保证输出速度，避免复杂的 UI 渲染导致闪烁
                    message_placeholder.markdown(full_response + "▌")

                # --- 关键点：生成完成后，清除原始内容，替换为漂亮的结构化显示 ---
                message_placeholder.empty()  # 清空占位符
                parse_and_render_message(full_response)  # 调用函数进行最终渲染

            except Exception as e:
                st.error(f"生成出错: {e}")
                full_response = "抱歉，系统遇到了一些问题，请稍后再试。"

    # 保存助手回答
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat_history(st.session_state.current_chat_id, st.session_state.messages)

# 样式微调
st.markdown("""
<style>
    .stButton>button {border-radius: 8px;}
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] button {
        text-align: left;
        border: 1px solid #eee;
    }
    /* 调整 Expander 的样式，让它看起来更像思考框 */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
        font-size: 0.9em;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)