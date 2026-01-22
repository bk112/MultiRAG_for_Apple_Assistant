# MultimodalRAG：多模态检索增强生成系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/singularguy/MultimodalRAG/actions)
[![Documentation](https://img.shields.io/badge/Docs-Latest-blue.svg)](https://jjrh0ec8rc.feishu.cn/docx/V5BrdafX1ovqL2xbiNlcDdsHnUh)
[![Code Coverage](https://img.shields.io/badge/Coverage-85%25-green.svg)](https://github.com/singularguy/MultimodalRAG)
[![Deepwiki Analysis](https://img.shields.io/badge/Deepwiki-AI%20Analysis-orange.svg)](https://deepwiki.com/singularguy/MultimodalRAG)
[![Issues](https://img.shields.io/github/issues/singularguy/MultimodalRAG.svg)](https://github.com/singularguy/MultimodalRAG/issues)
[![Stars](https://img.shields.io/github/stars/singularguy/MultimodalRAG?style=social)](https://github.com/singularguy/MultimodalRAG/stargazers)

## 📖 项目简介

本项目实现了一个**多模态检索增强生成（RAG）**系统，集成 **CLIP**、**Faiss** 和 **智谱 AI**，能够处理**文本**和**图像**数据，支持索引、检索以及生成文本、图像及多模态查询的响应。

✨ **关注我的分享** ✨  
- **小红书**：AnthroSeekTheX (Let's Seek The X!)  
- **技术方案与思考**：[飞书文档](https://jjrh0ec8rc.feishu.cn/docx/V5BrdafX1ovqL2xbiNlcDdsHnUh)

💡 **小提示** 💡  
想快速获取代码的 AI 解析？试试将 URL 中的 `github.com` 替换为 `deepwiki.com`（例如：`https://github.com/singularguy/MultimodalRAG` → `https://deepwiki.com/singularguy/MultimodalRAG`）。依赖 Deepwiki 服务。

---

## 🚀 核心功能

- **多模态索引**：同时索引文本描述和关联图像。  
- **向量嵌入**：使用 Hugging Face 的 CLIP 模型（`openai/clip-vit-base-patch32`）为文本和图像生成统一向量表示。  
- **高效检索**：采用 Faiss（`IndexIDMap2` + `IndexFlatIP`）实现快速相似性搜索。  
- **持久化存储**：通过 SQLite 存储文档元数据，并将 Faiss 索引保存至磁盘。  
- **上下文生成**：结合检索结果，使用智谱 AI（`glm-4-flash`）生成精准回答。  
- **灵活查询**：支持纯文本、纯图像及文本+图像的多模态查询。  
- **模块化设计**：代码分为 `MultimodalEncoder`、`Indexer`、`Retriever` 和 `Generator` 类，结构清晰。

---

## ⚙️ 系统要求

- **Python**：3.9 或更高版本  
- **智谱 AI API 密钥**：可从[智谱 AI 开放平台](https://open.bigmodel.cn/)获取  
- **Python 依赖**：见 `requirements.txt`

---

## 🛠️ 安装步骤

1. **克隆仓库**：
   ```bash
   git clone https://github.com/singularguy/MultimodalRAG.git
   cd MultimodalRAG
   ```

2. **创建虚拟环境**：
   ```bash
   # 使用 conda（推荐）
   conda create -n multimodal_rag python=3.12 -y
   conda activate multimodal_rag

   # 或使用 venv
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate   # Windows
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
   > **注意**：`requirements.txt` 包含 `faiss-cpu`。若需 GPU 支持，请配置 CUDA 后安装 `faiss-gpu`，并卸载 `faiss-cpu`。

---

## 🔑 配置

1. **设置智谱 AI API 密钥**：
   在项目根目录创建 `.env` 文件，添加：
   ```
   ZHIPUAI_API_KEY=your_api_key
   ```
   确保脚本能加载 `.env` 文件（例如使用 `python-dotenv` 库）。

---

## 🚀 运行系统

1. **准备数据**：参见下方[数据准备](#-数据准备)部分。  
2. **运行脚本**：
   ```bash
   python MultimodalRAG.py
   ```
   或使用 Jupyter 笔记本：
   ```bash
   jupyter notebook MultimodalRAG.ipynb
   ```

   脚本将自动：  
   - 加载数据和图像。  
   - 初始化编码器、索引器、检索器和生成器。  
   - 为文本、图像和平均向量构建索引。  
   - 执行示例查询并保存结果。  
   - 将索引和数据库持久化到磁盘。

---

## 📁 数据准备

1. **data.json**：
   JSON 文件，包含文档对象列表，每个对象需包含：  
   - `name`：唯一文档 ID（例如 `Bandgap1`）。  
   - `description`：文档的文本描述。  

   示例 `data.json`：
   ```json
   [
     {
       "name": "Bandgap1",
       "description": "一个基础的带隙基准电路图，展示 BJT 晶体管和电阻，用于生成温度不敏感的参考电压。"
     },
     {
       "name": "PTAT_Current",
       "description": "该原理图展示如何使用两个不匹配的 BJT 生成与绝对温度成正比（PTAT）的电流。"
     }
   ]
   ```

2. **images/ 目录**：
   存放与 `data.json` 中 `name` 字段对应的图像文件（例如 `Bandgap1.png`、`PTAT_Current.jpg`）。无图像的文档可不提供。因为我本地内容不方便泄露，放了个mock_data.json，大家可以参考。运行的时候改成data.json就行，里面是示例数据，需要替换成自己的数据。

---

## 🏗️ 代码结构

- **MultimodalEncoder**：使用 CLIP 将文本和图像编码为向量。  
- **Indexer**：管理 Faiss 索引和 SQLite 元数据存储。  
- **Retriever**：编码查询并检索相关文档。  
- **Generator**：与智谱 AI API 交互生成响应。  
- **主脚本**（`MultimodalRAG.py`）：协调整个 RAG 流程。

---

## 💡 注意事项与未来改进

- **性能优化**：大规模数据可使用 `faiss-gpu` 或高级 Faiss 索引（如 `IndexIVFFlat`）。  
- **多模态融合**：探索比简单向量平均更复杂的融合策略。  
- **图像理解**：升级到多模态 LLM（例如 `GLM-4V`）直接处理图像。  
- **错误处理**：增强健壮的错误管理和日志记录。  
- **可扩展性**：生产环境中可替换 SQLite 为向量数据库。  
- **提示工程**：优化生成器的 Prompt 以提升效果。  
- **文本分块**：对长文档进行分块处理。

---

## 📅 更新日志

- **2024-05-12**：精简项目结构，完善说明文档。  
- **2024-04-28**：优化 CLIP 文本/图像处理、存储和检索方法。  
- **2024-04-27**：添加多种技术方案的初步实现。  
- **2024-04-25**：初始化代码库。

---

## 📚 参考资料

- [苏子 Ansel OneRAG 系列](https://zhuanlan.zhihu.com/p/1890548455673422523)  
- [考拉的 AI 树屋（B 站同名）](https://space.bilibili.com/3546641711911609)  
- [MiniCPM-V 做多模态 RAG](https://modelbest.feishu.cn/wiki/NwhIwkJZYiHOPSkzwPUcq6hanif)  
- [多模态 RAG 的三种实现](https://zhuanlan.zhihu.com/p/1885630799371548044)  
- [智谱 TrustRAG](https://www.zhihu.com/question/6158926419/answer/65659770632)

---

## 🤝 如何贡献

欢迎任何形式的贡献！请：  
1. 通过 [Issues](https://github.com/singularguy/MultimodalRAG/issues) 报告问题或建议。  
2. 通过 [Pull Requests](https://github.com/singularguy/MultimodalRAG/pulls) 提交改进。

---

## 📄 许可证

本项目采用 [MIT 许可证](https://opensource.org/licenses/MIT) 开源。
