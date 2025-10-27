# app.py
#
# Streamlit UI
# - 상단에 🍐 이화 스타일 헤더
# - 파일 업로드
# - 질문 입력 / 답변 출력
#
# 실행: streamlit run app.py
#
# 주의: Streamlit Cloud 배포 시, requirements.txt에 있는 라이브러리 설치가 자동으로 이뤄져야 함.

import streamlit as st
from rag_core import RAGSessionState

# ---------- 세션 상태 초기화 ----------
if "rag_state" not in st.session_state:
    st.session_state["rag_state"] = RAGSessionState()

rag_state: RAGSessionState = st.session_state["rag_state"]

# ---------- 페이지 기본 설정 ----------
st.set_page_config(
    page_title="🍐 이화 RAG 도우미 ✿",
    page_icon="🍐",
    layout="wide",
)

# ---------- 간단한 CSS 커스터마이징 ----------
custom_css = """
<style>
/* 페이지 전체 배경 (연한 파스텔 톤) */
main {
    background-color: #f9fff4;
}

/* 상단 제목 영역 꾸미기 */
.ehwa-title {
    font-size: 2rem;
    font-weight: 700;
    color: #2e4f2e;
    background: #e8ffe8;
    padding: 12px 20px;
    border-radius: 16px;
    border: 2px solid #98c89b;
    display: inline-block;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    font-family: "Pretendard","Apple SD Gothic Neo",sans-serif;
}

/* 부제목 텍스트 */
.subtitle {
    color: #4a6b4a;
    font-size: 0.95rem;
    margin-top: 6px;
    margin-bottom: 18px;
    line-height: 1.4rem;
}

/* 업로더 / Q&A 박스 공통 스타일 */
.round-box {
    background: #ffffff;
    border-radius: 16px;
    border: 1.5px solid #cfe3cf;
    padding: 16px 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    margin-bottom: 1rem;
    font-family: "Pretendard","Apple SD Gothic Neo",sans-serif;
}

/* 답변 박스 (말풍선 느낌) */
.answer-bubble {
    background: #ffffff;
    border-radius: 16px;
    border: 2px solid #9ad09a;
    padding: 16px 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    white-space: pre-wrap;
    font-size: 0.95rem;
    line-height: 1.5rem;
    font-family: "Pretendard","Apple SD Gothic Neo",sans-serif;
    color: #2e2e2e;
}

/* 작은 라벨 스타일 */
.label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #3d603d;
    background: #eaffe0;
    border: 1px solid #a8d6a8;
    padding: 4px 8px;
    border-radius: 10px;
    display: inline-block;
    margin-bottom: 8px;
    font-family: "Pretendard","Apple SD Gothic Neo",sans-serif;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------- 헤더 영역 ----------
st.markdown(
    '<div class="ehwa-title">🍐 이화 RAG 도우미 ✿</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    '내가 올린 자료로만 대답해주는 개인 학습 챗봇입니다.<br>'
    '유료 OpenAI 결제가 없어도, 업로드한 파일 내용 기반으로 답변을 정리해 드려요 🍐<br>'
    '1) 파일 업로드 → 2) 질문 입력 → 3) 답변 확인 순서로 사용하세요.'
    '</div>',
    unsafe_allow_html=True
)

# ---------- 레이아웃 나누기 ----------
left_col, right_col = st.columns([1, 1])

# ---------- 왼쪽: 파일 업로드 영역 ----------
with left_col:
    st.markdown('<div class="round-box">', unsafe_allow_html=True)
    st.markdown('<div class="label">📂 자료 업로드</div>', unsafe_allow_html=True)
    st.write("여기에 공부 자료(PDF 텍스트 추출 가능한 버전, TXT, MD 등)를 업로드해 주세요.")
    st.write("업로드된 내용은 이 앱이 메모리로만 사용하고, 외부로 전송하지 않습니다.")

    uploaded_file = st.file_uploader(
        "파일을 선택하세요 (한 개씩 추가 업로드 가능)",
        type=["txt", "md", "csv", "py", "pdf"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        rag_state.add_document(file_bytes, uploaded_file.name)
        st.success(f"'{uploaded_file.name}' 업로드 완료! 🍐 벡터DB 갱신됐어요.")

    # 현재까지 몇 개 문서 반영됐는지 표시
    st.write(f"현재 반영된 문서 수: {len(rag_state.raw_texts)} 개")
    st.write(f"현재 생성된 청크 수: {len(rag_state.all_chunks)} 개")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- 오른쪽: 질문 / 답변 영역 ----------
with right_col:
    st.markdown('<div class="round-box">', unsafe_allow_html=True)
    st.markdown('<div class="label">💬 질문하기</div>', unsafe_allow_html=True)

    user_query = st.text_input(
        "궁금한 내용을 한국어로 입력하세요 🍐 예: '이 문서에서 핵심 개념이 뭐야?'",
        value=""
    )

    ask_button = st.button("질문 보내기 ✿")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="round-box">', unsafe_allow_html=True)
    st.markdown('<div class="label">📝 답변</div>', unsafe_allow_html=True)

    if ask_button:
        if user_query.strip() == "":
            st.warning("질문을 입력해 주세요 ✿")
        else:
            answer_text = rag_state.ask(user_query)
            st.markdown(
                f'<div class="answer-bubble">{answer_text}</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("질문을 입력하고 '질문 보내기 ✿' 버튼을 눌러주세요.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- 하단 안내 ----------
st.markdown("---")
st.markdown(
    """
    **사용 가이드 🍐**
    1. 왼쪽에서 파일을 업로드하면, 그 내용이 자동으로 지식베이스에 추가됩니다.
    2. 오른쪽에서 질문을 입력하면, 문서에서 가장 비슷한 부분을 찾아 정리해서 보여줍니다.
    3. 현재 버전은 OpenAI 유료 API 없이 '문서 기반 요약 답변' 형식으로 동작합니다.
    4. 즉, 일반 상식이 아니라 *업로드된 문서 내용*만을 근거로 답해요.
    """
)
