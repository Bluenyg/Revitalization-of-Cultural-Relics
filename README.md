# 🐴 国宝AI活化工作台：舞马衔杯仿皮囊式银壶

> 基于 Ollama (Qwen3-14B) + RAG 技术构建的"反人机交互"短视频创作辅助系统。

---

## 📖 项目简介

本项目属于"2025联合课程实践"项目的一部分,旨在通过 AI 技术赋能国宝文物的活化传播。

系统针对"舞马衔杯仿皮囊式银壶"这一文物,构建了一个专业的 RAG(检索增强生成)工作台。它能够辅助创作者完成《国宝画重点》栏目的短视频创意生成,特别是针对 STEP 3 阶段的"对话纠偏"任务。

### ✨ 核心功能

1. **多角色视角切换**:
   - 🧐 **专家学者**: 严谨考据史实与工艺
   - 🎨 **交互设计师**: 构思"反人机交互"的创新玩法(如重力感应、面部追踪)
   - 🔮 **符号学者**: 解读"胡汉融合"、"祝寿"等深层文化隐喻
   - 📢 **策展人**: 优化抖音平台的传播策略

2. **史实纠偏机制**: 基于本地向量知识库,自动识别并修正创意中的历史错误

3. **私有化部署**: 模型与数据完全运行在自有服务器上,保障数据安全

---

## 📂 项目结构

```text
project_root/
├── knowledge_base/          # 📚 知识库目录
│   └── silver_flask.md      # 舞马衔杯银壶的核心资料(Markdown格式)
├── chroma_db/               # 🗄️ 向量数据库(运行脚本后自动生成)
├── chat_history/            # 💾 对话记录自动保存目录
├── ingest.py                # ⚙️ 数据入库脚本(只需运行一次)
├── app.py                   # 🖥️ Web 应用主程序
├── requirements.txt         # 📦 依赖列表
└── README.md                # 📄 项目说明书
```

---

## 🚀 运行与维护指南

### 1.创建与激活环境
```bash
#创建环境
python -m venv venv

#激活环境
source venv/bin/activate
```

### 2. 启动应用 (后台挂载)

为了确保关闭 SSH 终端后应用继续运行,并允许局域网访问,请使用 `nohup` 命令启动:

```bash
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
```

**参数说明:**
- `--server.port 8501`: 指定运行端口
- `--server.address 0.0.0.0`: **关键参数**,允许外部 IP(如局域网内的同学)访问,而不仅仅是本机
- `> streamlit.log 2>&1`: 将所有输出日志保存在 `streamlit.log` 文件中,方便排查问题
- `&`: 让程序在后台运行

---

### 2. 查看运行状态

要查看 Streamlit 是否正在运行,或者查找它的进程号 (PID):

```bash
ps aux | grep streamlit
```

你将看到类似如下的输出,第二列的数字即为 PID:

```
wuzz 12345 1.0 2.5 ... python streamlit run app.py ...
```

---

### 3. 停止/重启应用

如果需要停止服务或重启应用,请先找到 PID,然后使用 `kill` 命令:

```bash
# 语法: kill -9 <PID>
kill -9 12345  # 将 12345 替换为你实际查到的进程号
```

---

### 4. 如何访问网页

#### 🏫 场景 A: 在学校/实验室 (局域网直连)

如果你和服务器连接的是同一个学校网络(WiFi 或有线),可以直接通过服务器内网 IP 访问:

👉 `http://<你的服务器IP>:8501`

**示例:** `http://10.122.202.53:8501`

---

#### 🌍 场景 B: 在校外 (外网访问)

如果在校外无法直接连接内网 IP,请配合 Cloudflare Tunnel 使用:

1. 确保 Streamlit 已按上述方式启动
2. 运行穿透工具:

```bash
./cloudflared-linux-amd64 tunnel --url http://localhost:8501
```

3. 使用生成的 `.trycloudflare.com` 链接访问

---

## 🛠️ 快速部署流程

### 步骤 1: 环境安装

```bash
pip install -r requirements.txt
```

### 步骤 2: 模型准备 (Ollama)

```bash
ollama pull qwen3:14b
ollama pull nomic-embed-text
```

### 步骤 3: 构建知识库

```bash
python3 ingest.py
```

### 步骤 4: 启动服务

参考上方 **"启动应用"** 章节执行启动命令

---

## 📝 注意事项

- 首次运行 `ingest.py` 时会创建向量数据库,可能需要几分钟时间
- 对话记录会自动保存在 `chat_history/` 目录下
- 如遇到端口占用,可修改启动命令中的端口号(如改为 8502)
- 日志文件 `streamlit.log` 可用于调试和监控应用状态

---

## 🤝 技术支持

如有问题,请检查:
1. Ollama 服务是否正常运行: `ollama list`
2. 所需模型是否已下载
3. 防火墙是否开放 8501 端口(局域网访问时)