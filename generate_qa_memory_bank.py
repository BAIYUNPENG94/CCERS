import json
import os
import re
import sys
from collections import defaultdict

from FlagEmbedding import FlagModel
from tqdm import tqdm

sys.path.append(os.getcwd())

from utils.config import BGEModel_PATH
from utils.functions import call_llm

system_prompt = """
You're a master of emotional analysis, and you're able to carefully discern the emotional state of each conversational character.
It is assumed that each character has a total of 8 basic emotions, including joy, acceptance, fear, surprise, sadness, disgust, anger, and anticipation. 
Next I will enter a conversation with {role} and your task is to analyze {role}'s score on these 8 emotion dimensions, with a minimum of 1 and a maximum of 10, with higher scores indicating that {role} expresses this emotion dimension more strongly.
Please analyze the performance of {role} on the 8 sentiment dimensions, give the scoring reasons and scores, and finally output the results as a python list as follows:
[
    {{"analysis": <REASON>, "dim": "joy", "score": <SCORE>}},
    {{"analysis": <REASON>, "dim": "acceptance", "score": <SCORE>}},
    ...
    {{"analysis": <REASON>, "dim": "anticipation", "score": <SCORE>}}
]
Your answer must be a valid python list to ensure that I can parse it directly using python, without redundancy! Please give results that are as accurate as possible and in line with most people's intuition.
"""

embedding_model = FlagModel(
    BGEModel_PATH,
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True,
)


def str2vector(text):
    match = re.findall(r"【(.*?)】", text)
    emotion_str = match[-1]

    emotion_str = emotion_str.replace(": ", "：")
    emotion_str = emotion_str.replace(":", "：")
    emotion_str = emotion_str.replace(", ", "，")
    emotion_str = emotion_str.replace(",", "，")

    emotion_list = emotion_str.split("，")
    emotion_embedding = [float(e.split("：")[1]) for e in emotion_list]
    return emotion_embedding

def parse_emotion_response(response):
    try:
        # 🧼 1. 先尝试直接解析（万一它真的是合法 JSON）
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 🧼 2. 如果报错，尝试手动修复格式（用正则强行提取 JSON）
    try:
        # 找所有 {"analysis": "...", "dim": "...", "score": x} 的结构
        matches = re.findall(r'\{[^{}]*"dim"\s*:\s*"[^"]+",\s*"score"\s*:\s*\d+[^{}]*\}', response)
        if not matches:
            print("没匹配到任何类似 JSON 的结构")
            return []

        cleaned = []
        for match in matches:
            try:
                item = json.loads(match)
                cleaned.append({"dim": item["dim"], "score": item["score"]})
            except:
                continue
        return cleaned
    except Exception as e:
        print("清洗失败：", e)
        return []


def get_result(data):
    context_embedding = embedding_model.encode(data["context"]).tolist()

    messages = [
        {"role": "system", "content": system_prompt.format(role=data["role"])},
        {"role": "user", "content": data["context"]},
    ]
    response, _ = call_llm("llama3", messages)  # analyze sentiment using entire conversation history
    try:
#        emotion_list = json.loads(response)
        emotion_list = parse_emotion_response(response)
        print(emotion_list)
    except:
        print('gpt reply format error!')
        print(response)
        return None
    emotions = [
        "joy",
        "acceptance",
        "fear",
        "surprise",
        "sadness",
        "disgust",
        "anger",
        "anticipation",
    ]
    emotion_embedding = [1, 1, 1, 1, 1, 1, 1, 1]
    for item in emotion_list:
#       if item["dim"] in emotions:
        dim = item.get("dim")
        score = item.get("score")
        if dim in emotions and isinstance(score, (int, float)):
            emotion_embedding[emotions.index(dim)] = score
#           emotion_embedding[emotions.index(item["dim"])] = item["score"]
    result = {
        "id": data["id"],
        "context": data["context"],
        "gpt_output": "response",
        "emotion_embedding": emotion_embedding,
        "context_embedding": context_embedding,
    }

    return result


with open("data//charactereval//test_data.jsonl", "r", encoding="utf-8") as f:#
    datas = json.load(f)

# Extract QA pairs
res_dict = defaultdict(list)
for data in datas:
    role = data["role"]
    dialogue_list = data["context"].split("\n")
    for i in range(1, len(dialogue_list)):
        speaker = dialogue_list[i].split("：")[0]
        if speaker == role:
            qa_pair = dialogue_list[i - 1] + "\n" + dialogue_list[i]
            memory_fragment = {"id": data["id"], "role": role, "context": qa_pair}
            res_dict[role].append(memory_fragment)

all_memeory_bank = {}

for role in res_dict:
    if role == "佟湘玉":
        role_memory_bank = [get_result(data) for data in tqdm(res_dict[role][:10])]
        all_memeory_bank[role] = role_memory_bank

with open("data/charactereval/all_memory_bank.jsonl", "w", encoding="utf-8") as f:
    json.dump(all_memeory_bank, f, ensure_ascii=False, indent=2)
