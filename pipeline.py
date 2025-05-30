from supertoneAPi import generate_tts
from clovaSpeechApi import clova_short_stt
from rag import chunk_text, build_faiss, retrieve_top_k
from gpt import generate_answer, save_answer

if __name__ == "__main__":
    audio_path = "C:/Users/sink0/캡스톤/Van.wav"
    text_path = "반고흐_편지_원문.txt"

    # 1. STT → 질문 추출
    print("start STT")
    question = clova_short_stt(audio_path)
    if not question:
        print("STT 실패")
        exit()
    print("complete STT")

    # 2. 텍스트 불러오기 및 청크
    print("start ChatGPT")
    with open(text_path, encoding="utf-8") as f:
        full_text = f.read()
    chunks = chunk_text(full_text)

    # 3. FAISS 인덱싱
    index, _ = build_faiss(chunks)

    # 4. GPT 기반 RAG 응답 생성
    context = retrieve_top_k(index, question, chunks)
    answer = generate_answer(context, question)

    # 5. 텍스트 저장
    path = save_answer(answer)
    print("답변 저장:", path)
    print("complete ChatGPT")

    # 6. TTS 변환 (텍스트가 너무 길면 자르기)
    print("start TTS")
    if len(answer) > 200:
        print("TTS 텍스트가 200자를 초과하여 자릅니다.")
        answer = answer[:200]
    generate_tts(answer, output_path="van_gogh_reply.wav")
    print("complete TTS")
