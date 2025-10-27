# rag_core.py
#
# RAG 핵심 로직:
# - 업로드된 문서를 텍스트로 추출/분할
# - 임베딩으로 벡터DB 생성 (Chroma)
# - 사용자 질문과 가장 비슷한 문서조각을 찾아서 답변 후보 생성
#
# OpenAI 결제 없이 동작하도록 sentence-transformers 기반 임베딩 사용.

import io
from typing import List, Tuple
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_file_to_text(file_bytes: bytes, filename: str) -> str:
    """
    업로드된 파일을 텍스트로 변환하는 간단한 함수.
    현재는 .txt / .md / .csv / .py 등 '텍스트 기반' 파일 위주로 처리.
    PDF 등 복잡한 포맷은 여기서 확장 가능.
    """
    lower_name = filename.lower()
    # 가장 단순한 방식: 그냥 utf-8 디코딩 시도
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        return text
    except Exception:
        pass

    # 혹시 모를 다른 인코딩
    try:
        text = file_bytes.decode("cp949", errors="ignore")
        return text
    except Exception:
        pass

    # 마지막 fallback: 바이너리 -> 빈 문자열
    return ""


def split_text_to_chunks(text: str,
                         chunk_size: int = 500,
                         chunk_overlap: int = 100) -> List[str]:
    """
    긴 텍스트를 작은 청크로 나누기.
    chunk_size / chunk_overlap 값은 너무 공격적으로 키우지 않음 (학생 난이도).
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks


def build_vectorstore_from_chunks(chunks: List[str]):
    """
    청크 리스트를 받아서 Chroma 벡터스토어를 메모리 상에 생성.
    HuggingFace 임베딩 모델 사용 → 무료.
    모델은 소형 SBERT 계열을 사용해 학생 환경에서도 비교적 가볍게 동작.
    """
    if not chunks:
        return None

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model
    )
    return vectorstore


def retrieve_similar_passages(
    vectorstore,
    query: str,
    k: int = 3
) -> List[Tuple[str, float]]:
    """
    사용자의 질문(query)과 유사한 상위 k개 문단을 검색.
    반환: [(문단내용, 유사도 점수), ...]
    """
    if vectorstore is None:
        return []

    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)

    results = []
    for doc, score in docs_and_scores:
        results.append((doc.page_content, score))
    return results


def build_answer_from_passages(query: str,
                               passages: List[Tuple[str, float]]) -> str:
    """
    OpenAI 유료 API 없이 답변을 "생성"하는 방식.
    - 상위 관련 문단들을 뽑아서 요약 형태로 보여준다.
    - '근거 기반 답변'처럼 보이도록 구성.
    """
    if not passages:
        return (
            "📘 관련 내용을 찾지 못했습니다.\n"
            "업로드한 문서에 해당 질문과 유사한 내용이 거의 없거나\n"
            "아직 문서를 업로드하지 않았을 수 있어요. 🍐"
        )

    answer_lines = []
    answer_lines.append("🍐 질문: " + query.strip())
    answer_lines.append("")
    answer_lines.append("📚 문서에서 찾은 관련 내용 요약:")

    for idx, (text_block, score) in enumerate(passages, start=1):
        # 너무 긴 블록을 한 번 더 잘라서 깔끔하게
        short_preview = text_block.strip()
        if len(short_preview) > 400:
            short_preview = short_preview[:400] + " ..."

        answer_lines.append(f"\n[{idx}] {short_preview}")

    answer_lines.append(
        "\n✿ 위 내용은 업로드된 문서에서 직접 검색된 근거입니다.\n"
        "✿ 즉, 이 챗봇은 일반적인 지식이 아니라 '내 자료'를 기반으로 답해요.\n"
    )

    return "\n".join(answer_lines)


class RAGSessionState:
    """
    Streamlit 세션과 연결해서 쓸 작은 상태 관리용 클래스.
    - 업로드 문서 전체 텍스트
    - 잘라낸 청크
    - 벡터스토어
    """
    def __init__(self):
        self.raw_texts = []        # 원본 텍스트들 (파일별)
        self.all_chunks = []       # 잘린 청크 전체
        self.vectorstore = None    # Chroma 벡터스토어

    def add_document(self, file_bytes: bytes, filename: str):
        """
        새 파일을 세션에 추가하고, 전체 벡터스토어를 다시 빌드한다.
        (간단하게 '덮어쓰기' 식으로 재구성)
        """
        text = load_file_to_text(file_bytes, filename)
        if text.strip():
            self.raw_texts.append(text)

        # 모든 문서를 합쳐서 다시 청크화
        merged = "\n\n".join(self.raw_texts)
        chunks = split_text_to_chunks(merged)
        self.all_chunks = chunks
        self.vectorstore = build_vectorstore_from_chunks(chunks)

    def ask(self, query: str) -> str:
        """
        사용자의 질문에 대해 RAG 검색 후 요약형 답변 생성.
        """
        passages = retrieve_similar_passages(self.vectorstore, query)
        answer = build_answer_from_passages(query, passages)
        return answer
