# rag_core.py (TF-IDF 轻量版，无需HuggingFace和torch)
#
# 思路：
# 1. 把上传的文档切成chunk
# 2. 用 TfidfVectorizer 把所有chunk向量化
# 3. 用户提问 -> 也向量化 -> 计算余弦相似度
# 4. 取最相似的片段，组成回答
#
# 这样依然是“基于我上传的文档回答”，符合RAG逻辑，
# 而且不用huggingface模型，所以Streamlit Cloud不会报ImportError。

from typing import List, Tuple
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters import CharacterTextSplitter


def load_file_to_text(file_bytes: bytes, filename: str) -> str:
    """
    简单读取文本型文件。对PDF等复杂格式暂时不做OCR，只尝试直接decode。
    """
    lower_name = filename.lower()

    # 直接按utf-8读
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        return text
    except Exception:
        pass

    # 备用编码
    try:
        text = file_bytes.decode("cp949", errors="ignore")
        return text
    except Exception:
        pass

    return ""


def split_text_to_chunks(text: str,
                         chunk_size: int = 500,
                         chunk_overlap: int = 100) -> List[str]:
    """
    把长文本切成小段，保留一定重叠，方便检索。
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks


class SimpleVectorStore:
    """
    一个很轻量的向量库：
    - 用 TF-IDF 把所有chunk编码成向量矩阵
    - 做余弦相似度检索
    """

    def __init__(self, chunks: List[str]):
        self.chunks = chunks  # 文本片段列表
        self.vectorizer = TfidfVectorizer()
        if chunks:
            self.matrix = self.vectorizer.fit_transform(chunks)  # shape: (num_chunks, vocab_dim)
        else:
            self.matrix = None

    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        返回与query最相似的k个chunk，附带相似度分数。
        """
        if (self.matrix is None) or (not query.strip()):
            return []

        q_vec = self.vectorizer.transform([query])  # shape: (1, vocab_dim)

        # 余弦相似度 = (A · B) / (||A||*||B||)
        # 这里使用稀疏矩阵乘法得到点积，再除以范数
        dot_scores = (self.matrix @ q_vec.T).toarray().ravel()  # (num_chunks,)
        doc_norms = np.linalg.norm(self.matrix.toarray(), axis=1) + 1e-10
        q_norm = np.linalg.norm(q_vec.toarray()) + 1e-10
        cosine_scores = dot_scores / (doc_norms * q_norm)

        # 排序，取top k
        idx_sorted = np.argsort(cosine_scores)[::-1]  # 从大到小
        top_idx = idx_sorted[:k]

        results = []
        for i in top_idx:
            results.append((self.chunks[i], float(cosine_scores[i])))

        return results


def build_vectorstore_from_chunks(chunks: List[str]):
    """
    使用SimpleVectorStore，而不是HuggingFaceEmbeddings+Chroma。
    """
    if not chunks:
        return None
    return SimpleVectorStore(chunks)


def build_answer_from_passages(query: str,
                               passages: List[Tuple[str, float]]) -> str:
    """
    根据检索到的片段，组合一个"基于你的文档"的回答。
    不调用OpenAI，完全免费。
    """
    if not passages:
        return (
            "📘 没找到和问题强相关的内容。\n"
            "可能还没有成功解析这个文件，或者文档内容和提问差距太大。🍐"
        )

    answer_lines = []
    answer_lines.append("🍐 你的问题： " + query.strip())
    answer_lines.append("")
    answer_lines.append("📚 根据你上传的文档，最相关的内容是：")

    for idx, (text_block, score) in enumerate(passages, start=1):
        preview = text_block.strip()
        if len(preview) > 400:
            preview = preview[:400] + " ..."
        answer_lines.append(f"\n[{idx}] 相似度 {score:.3f}\n{preview}")

    answer_lines.append(
        "\n✿ 提示：以上回答只来自你上传的资料（本地检索），"
        "并不是互联网通用知识。\n"
    )

    return "\n".join(answer_lines)


class RAGSessionState:
    """
    保存会话状态：
    - 所有原始文本
    - 切分后的chunks
    - 一个TF-IDF向量库
    """
    def __init__(self):
        self.raw_texts = []
        self.all_chunks = []
        self.vectorstore = None

    def add_document(self, file_bytes: bytes, filename: str):
        """
        添加新文件后，重新构建chunks和向量库（简单粗暴版，足够学生作业）。
        """
        text = load_file_to_text(file_bytes, filename)
        if text.strip():
            self.raw_texts.append(text)

        merged = "\n\n".join(self.raw_texts)
        chunks = split_text_to_chunks(merged)
        self.all_chunks = chunks
        self.vectorstore = build_vectorstore_from_chunks(chunks)

    def ask(self, query: str) -> str:
        if self.vectorstore is None:
            return "还没有可检索的内容，请先上传文档 🍐"
        passages = self.vectorstore.similarity_search(query, k=3)
        answer = build_answer_from_passages(query, passages)
        return answer
