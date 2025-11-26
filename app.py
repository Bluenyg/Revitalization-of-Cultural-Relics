import os

# --- 0. 基础配置与环境设置 ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ALLOW_RESET"] = "True"

import streamlit as st
import json
import glob
import re
import time
import uuid
from datetime import datetime

# LangChain & RAG 库
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate  # [修改] 引入 PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFPlumberLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

# --- 页面配置 ---
st.set_page_config(
    page_title="国宝AI活化工作台 (Pro Max版)",
    page_icon="🏺",
    layout="wide"
)

# 定义路径
CHROMA_DB_DIR = "./chroma_db"
HISTORY_DIR = "./chat_history"
KNOWLEDGE_BASE_DIR = "./knowledge_base"
STATUS_FILE = "./db_status.json"
OLLAMA_URL = "http://localhost:11434"

# 确保目录存在
for directory in [HISTORY_DIR, CHROMA_DB_DIR, KNOWLEDGE_BASE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- 1. 深度角色定义 (保持不变) ---
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


# [新增] 格式化历史记录函数
def format_chat_history(messages, k=6):
    """将最近的 k 条对话记录转换为字符串，供模型理解上下文"""
    recent_msgs = messages[-k:]  # 只取最近k条，避免上下文过长
    history_text = ""
    for msg in recent_msgs:
        role = "用户" if msg["role"] == "user" else "AI助手"
        content = msg["content"].replace("<think>", "").replace("</think>", "")  # 清理think标签，减少干扰
        history_text += f"{role}: {content}\n"
    return history_text


# --- 状态管理函数 ---
def load_db_status():
    """读取已学习文件列表"""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except:
            return set()
    return set()


def update_db_status(filename, action="add"):
    """更新状态文件"""
    current_status = load_db_status()
    if action == "add":
        current_status.add(filename)
    elif action == "remove":
        if filename in current_status:
            current_status.remove(filename)

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(current_status), f, ensure_ascii=False, indent=4)


def get_vector_store():
    """获取向量数据库实例"""
    embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)


def ingest_file(filename):
    """将指定文件(已在文件夹中)存入向量库"""
    file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
    if not os.path.exists(file_path):
        return False, "文件不存在"

    try:
        # 1. 加载文件
        suffix = os.path.splitext(filename)[1].lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = TextLoader(file_path, autodetect_encoding=True)

        docs = loader.load()

        # 2. 切分文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        if not chunks:
            return False, "⚠️ 文档切分后内容为空。"

        # 添加元数据
        abs_path = os.path.abspath(file_path)
        for chunk in chunks:
            chunk.metadata['source'] = abs_path

        ids = [str(uuid.uuid4()) for _ in chunks]

        # 3. 存入数据库
        vector_store = get_vector_store()
        vector_store.add_documents(chunks, ids=ids)
        vector_store.persist()

        update_db_status(filename, "add")

        return True, f"✅ 已学习：{len(chunks)} 个知识块"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"处理失败: {str(e)}"


def delete_document_complete(filename):
    """彻底删除：删向量 + 删文件 + 删状态"""
    try:
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        abs_path = os.path.abspath(file_path)

        if filename in load_db_status():
            vector_store = get_vector_store()
            vector_store._collection.delete(where={"source": abs_path})
            vector_store.persist()
            update_db_status(filename, "remove")

        if os.path.exists(file_path):
            os.remove(file_path)
            return True, f"🗑️ 已彻底移除: {filename}"
        else:
            return True, f"⚠️ 文件已从库中移除，但磁盘上未找到: {filename}"

    except Exception as e:
        return False, f"删除失败: {str(e)}"


# 历史记录管理函数
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

    # === 模块 A: 知识库全流程管理 ===
    with st.expander("📚 资料库管理", expanded=True):
        uploaded_file = st.file_uploader("上传新资料", type=["pdf", "md", "txt"])
        if uploaded_file:
            save_path = os.path.join(KNOWLEDGE_BASE_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.toast(f"文件 {uploaded_file.name} 已保存，请点击'学习'入库。", icon="💾")
            time.sleep(1)
            st.rerun()

        st.divider()
        st.caption("📂 资料库列表")

        if os.path.exists(KNOWLEDGE_BASE_DIR):
            all_files = sorted(os.listdir(KNOWLEDGE_BASE_DIR))
        else:
            all_files = []

        learned_status = load_db_status()

        if not all_files:
            st.info("暂无文档")
        else:
            for filename in all_files:
                col_icon, col_name, col_btn = st.columns([0.15, 0.65, 0.2])
                is_learned = filename in learned_status

                with col_icon:
                    st.write("✅" if is_learned else "⚪")

                with col_name:
                    st.text(filename)

                with col_btn:
                    if not is_learned:
                        if st.button("学习", key=f"learn_{filename}", help="点击入库"):
                            with st.spinner("正在学习..."):
                                success, msg = ingest_file(filename)
                                if success:
                                    st.toast(msg, icon="🎉")
                                    st.rerun()
                                else:
                                    st.error(msg)
                    else:
                        if st.button("🗑️", key=f"del_{filename}", help="彻底删除"):
                            with st.spinner("正在清理..."):
                                success, msg = delete_document_complete(filename)
                                if success:
                                    st.toast(msg, icon="👋")
                                    st.rerun()
                                else:
                                    st.error(msg)

            unlearned_count = len([f for f in all_files if f not in learned_status])
            if unlearned_count > 0:
                st.divider()
                if st.button(f"🚀 一键学习剩余 {unlearned_count} 个文件", type="primary"):
                    progress_bar = st.progress(0)
                    for i, fname in enumerate(all_files):
                        if fname not in learned_status:
                            ingest_file(fname)
                        progress_bar.progress((i + 1) / len(all_files))
                    st.success("全部入库完成！")
                    time.sleep(1)
                    st.rerun()

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

if os.path.exists(CHROMA_DB_DIR):
    vector_store = get_vector_store()
    try:
        cnt = vector_store._collection.count()
        k = min(4, cnt) if cnt > 0 else 0
    except:
        k = 0

    if k > 0:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
    else:
        retriever = None
else:
    retriever = None

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            parse_and_render_message(msg["content"])
        else:
            st.markdown(msg["content"])

role_config = ROLE_DEFINITIONS[selected_role]

# [修改] 定义查询重写 (Contextualize) 的 Prompt
# 目的：将用户的 "它"、"那个" 等代词替换为历史中的具体名词
rephrase_prompt_template = """
给定以下对话历史和用户的最新问题，请将用户的最新问题改写为一个**独立、完整、不依赖上下文即可理解的问题**。
如果用户的问题已经很完整，直接返回原问题。
不要回答问题，只负责改写。不要输出任何思考过程或标签。

对话历史：
{chat_history}

用户最新问题：{question}

独立问题：
"""
rephrase_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=rephrase_prompt_template
)

# [修改] 定义最终回答的 Prompt (加入 chat_history)
system_template = f"""
你现在的身份是：**{selected_role}**。
{role_config['description']}

**你的核心关注点：**
{role_config['focus']}

**回复指导原则：**
{role_config['instruction']}

**专业分析维度参考：**
{PROFESSIONAL_ANGLES}

请根据以下【参考资料】和【对话历史】来回答用户的【最新问题】。
如果资料中没有答案，请运用你的专业知识进行合理推演，但必须声明这是推演。

---
【对话历史】：
{{chat_history}}

【参考资料】：
{{context}}
---
【最新问题】：
{{question}}
"""

final_prompt = ChatPromptTemplate.from_template(system_template)

if user_input := st.chat_input("请输入关于文物的创意或问题..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    # 在生成前先不加入 session_state，等生成完再加，或者现在加也可以，
    # 这里为了保持逻辑一致，我们手动维护给模型的 history，不包含当前这句 user_input

    # 1. 准备历史记录文本
    history_text = format_chat_history(st.session_state.messages, k=6)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner(f"{selected_role} 正在思考..."):

                # --- [步骤 1]: 上下文理解与查询重写 ---
                # 如果有历史记录，先进行重写；如果是第一句话，直接用原话
                actual_query = user_input
                if len(st.session_state.messages) > 1:
                    rephrase_chain = rephrase_prompt | llm | StrOutputParser()
                    reformulated_question = rephrase_chain.invoke({
                        "chat_history": history_text,
                        "question": user_input
                    })
                    # 清理可能产生的多余空白
                    actual_query = reformulated_question.strip()

                    # 调试信息：展示重写后的问题（可选，觉得不需要可以注释掉）
                    with st.expander("🔍 上下文理解 (查询重写)", expanded=False):
                        st.write(f"原问题: {user_input}")
                        st.write(f"理解为: {actual_query}")

                # --- [步骤 2]: 检索 ---
                context_text = ""
                if retriever:
                    # 使用重写后的问题去检索
                    docs = retriever.get_relevant_documents(actual_query)
                    context_text = "\n\n".join([d.page_content for d in docs])

                if not context_text:
                    context_text = "暂无本地知识库相关内容，请依靠你的通用知识回答。"

                # --- [步骤 3]: 生成回答 ---
                # 将 重写后的问题(或者原问题) + 检索到的上下文 + 历史记录 传给最终 Prompt
                # 注意：这里我们通常把 user_input 传给 Prompt 显示给用户看，
                # 但实际上 retrieve 用的是 actual_query。
                # 有一种做法是 Prompt 里也放 actual_query，但为了保持对话自然，
                # 我们Prompt里还是放 user_input，因为上下文都在 chat_history 里了，
                # 主要是为了让 Context (参考资料) 是准确的。

                chain = (
                        final_prompt
                        | llm
                        | StrOutputParser()
                )

                # 流式输出
                stream_input = {
                    "chat_history": history_text,
                    "context": context_text,
                    "question": user_input  # 这里用原话，因为Prompt里有History兜底，且Retriever已经用actual_query找过资料了
                }

                for chunk in chain.stream(stream_input):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

                placeholder.empty()
                parse_and_render_message(full_response)

        except Exception as e:
            import traceback

            traceback.print_exc()
            st.error(f"生成出错: {e}")
            full_response = "抱歉，系统出了点小差错。"

    # 更新 Session State
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat_history(st.session_state.current_chat_id, st.session_state.messages)