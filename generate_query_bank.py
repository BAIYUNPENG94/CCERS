import json
import os
import re
import sys

from FlagEmbedding import FlagModel
from tqdm import tqdm
import boto3

sys.path.append(os.getcwd())

from utils.config import BGEModel_PATH
from utils.functions import call_llm

system_prompt = """
You're a master of emotional analysis, and you're able to carefully discern the emotional state embedded in the interviewer's questions。
It is assumed that there are eight basic emotions, including joy, acceptance, fear, surprise, sadness, disgust, anger, and anticipation.
Next I will enter an interviewer's question and your task is to analyze how the question scores on these 8 emotional dimensions, with a minimum of 1 and a maximum of 10, with higher scores indicating that the question is more strongly expressed on this emotional dimension.
Please analyze the performance of the question on the 8 sentiment dimensions, give the scoring rationale and score, and finally output the results as a python list as shown below:
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
            print("⚠️ 没匹配到任何类似 JSON 的结构")
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
        print("🚨 清洗失败：", e)
        return []


def get_result(data):
    context_embedding = embedding_model.encode(data["context"]).tolist()

#    context_embedding = get_titan_embedding(""amazon.titan-embed-text-v1"", data["context"], region)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["context"]},
    ]
#    response, _ = call_llm("gpt-3.5-turbo", messages)  # analyze sentiment using entire conversation history
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
#        if item["dim"] in emotions:
        dim = item.get("dim")
        score = item.get("score")
        if dim in emotions and isinstance(score, (int, float)):
            emotion_embedding[emotions.index(dim)] = score
#            emotion_embedding[emotions.index(item["dim"])] = item["score"]

    data["emotion_embedding"] = emotion_embedding
    data["context_embedding"] = context_embedding

    return data


with open("data//16Personalities.json", "r", encoding="utf-8") as f:
    datas = json.load(f)

query_bank_16P_zh = []
query_bank_16P_en = []
for id, question in tqdm(datas["questions"].items()):
    query_zh = {"id": id, "context": question["rewritten_zh"]}
    response_zh = get_result(query_zh)
    if response_zh:
        query_bank_16P_zh.append(response_zh)
    query_en = {"id": id, "context": question["rewritten_en"]}
    response_en = get_result(query_en)
    if response_en:
        query_bank_16P_en.append(response_en)

with open("data/query_bank_16P_zh.jsonl", "w", encoding="utf-8") as f:
    json.dump(query_bank_16P_zh, f, ensure_ascii=False, indent=2)

with open("data/query_bank_16P_en.jsonl", "w", encoding="utf-8") as f:
    json.dump(query_bank_16P_en, f, ensure_ascii=False, indent=2)
