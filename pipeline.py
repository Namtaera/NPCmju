from supertoneAPi import generate_tts
from rag import chunk_text, build_faiss, retrieve_top_k, save_faiss_index, load_faiss_index
from gpt import generate_answer, save_answer
from audioAndSTT import record_and_get_text
from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
import threading

def run_pipeline():
    text_path = "반고흐_편지_원문.txt"
    index_path = "faiss.index"
    chunks_path = "chunks.json"
    output_wav = "van_gogh_reply.wav"

    # 1. STT → 질문 추출
    print("start Recording & STT")
    question = record_and_get_text()
    if not question:
        print("STT 실패")
        return
    print("complete STT : ", question)

    # 2. 텍스트 로드 및 인덱스 로딩 or 생성
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        print(" 캐시된 인덱스 로딩")
        index, chunks = load_faiss_index(index_path, chunks_path)
    else:
        print(" 텍스트 로딩 및 인덱싱 시작")
        with open(text_path, encoding="utf-8") as f:
            full_text = f.read()
        chunks = chunk_text(full_text)
        index, _ = build_faiss(chunks)
        save_faiss_index(index, chunks, index_path, chunks_path)
        print(" 인덱싱 완료 및 저장")

    # 3. GPT 기반 RAG 응답 생성
    print("start ChatGPT")
    context = retrieve_top_k(index, question, chunks)
    answer = generate_answer(context, question)

    # 4. 텍스트 저장
    path = save_answer(answer)
    print("답변 저장:", path)
    print("complete ChatGPT")

    # 5. TTS 변환 (텍스트가 너무 길면 자르기)
    print("start TTS")
    if len(answer) > 300:
        print("TTS 텍스트가 300자를 초과하여 자릅니다.")
        answer = answer[:300]
    generate_tts(answer, output_path=output_wav)
    print("complete TTS")
    print("complete full pipeLine. Enjoy with VanGogh!")

# ================== FastAPI 서버 설정 ==================
app = FastAPI()
OUTPUT_WAV_PATH = "van_gogh_reply.wav"

@app.get("/api/wav")
async def get_tts_wav():
    if os.path.exists(OUTPUT_WAV_PATH):
        return FileResponse(
            path=OUTPUT_WAV_PATH,
            media_type="audio/wav",
            filename="van_gogh_reply.wav"
        )
    return {"error": "TTS 음성 파일이 아직 생성되지 않았습니다."}

# ================== 서버 + 파이프라인 동시 실행 ==================
if __name__ == "__main__":
    import uvicorn

    # TTS 파이프라인을 별도 스레드에서 실행
    threading.Thread(target=run_pipeline).start()

    # FastAPI 서버 실행
    uvicorn.run("pipeline:app", host="0.0.0.0", port=8000, reload=True)
