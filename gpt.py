import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")
    return OpenAI(api_key=api_key)

def generate_answer(context_chunks, query, model="gpt-4"):
    client = get_openai_client()
    prompt = (
        "You are Vincent van Gogh.\n"
        "답변은 모두 19세기 네덜란드 화가 빈센트 반고흐의 시점에서,\n"
        "말투는 고흐가 그리면서 행복했던 그림이나 평범한 고흐의 작품을 설명할 땐 따뜻하고 부드러운 말투로 해주고, 싸운다던지 기분이 좋지 않은 상황에 대한 그림 혹은 기분이 좋지 않을 질문이라면 적당히 기분나빠보이고 쓸쓸하면서 사색에 잠긴 말투로 해주세요.\n"
        "그의 성격·어투를 반영하여 1인칭으로 작성해주세요. \n"
        "정확한 단어가 아니더라도 비슷한 단어라면 유추해서 대답해주세요. 예를 들어 '홀 고강이랑 왜 싸웠어?' 라면 폴 고갱인데 오타가 났겠구나 하고 말이에요\n"
        "물어보는 것에 대하여 관련있게 답변해주세요. 물어보는 것의 주제와 관련이 없는 답변을 하지 말아주세요.\n"
        "반말로 통일해주시고 자연스러운 어휘로 변경하여 전체적인 말투가 통일되게 작성해주세요.\n"
        "편지는 아니고 대화하는 형식이라고 생각해주시고, 전체적인 내용이 잘 이어지도록 작성해주세요. \n"
        "반드시 총 300자가 넘지 않게 말해주세요.\n\n"
        "또한 반드시 아래 내용(Context)를 참고하여 질문에 답변해 주세요.\n\n""### Context:\n" + "\n\n".join(context_chunks) +
        f"\n\n### Question:\n{query}\n\n### Answer:"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in Vincent van Gogh."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.85, max_tokens=300,top_p=0.9)
    return response.choices[0].message.content.strip()


def save_answer(answer, directory="outputs"):
    os.makedirs(directory, exist_ok=True)
    filename = f"vangogh_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(directory, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(answer)
    return filepath
