from openai import OpenAI
from copy import deepcopy
import numpy as np
import pandas as pd
import argparse

def query_gpt(query):
    chat_completion = client.chat.completions.create(
        messages=[
        {
            "role": "system",
            "content": """Please act as an impartial judge and evaluate the quality of the response provided by an
                        AI assistant to the user question displayed below. Your evaluation should consider factors
                        such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
                        the response. Begin your evaluation by providing a short explanation. Be as objective as
                        possible. After providing your explanation, please rate the response on a scale of 1 to 10
                        by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".""",
        },
        {
            "role": "user",
            "content": query
        }
        ],
        model="gpt-4",
        max_tokens=1000
    )
    completion = chat_completion.choices[0].message.content
    return completion

def get_questions(q_file):
    f = open(q_file,'r')
    data = f.read()
    all_data = data.split('\n')
    return all_data

def get_answers(q_file):
    df = pd.read_csv(q_file)
    all_data = df.values[:,1].tolist()
    return all_data

def get_scores_for_qa_pairs(questions, answers, prompt, save_file):
    scores = []
    completions = []
    for q,a in zip(questions, answers):
        prompt_temp = deepcopy(prompt)
        prompt_temp = prompt_temp.replace("{question}", q)
        prompt_temp = prompt_temp.replace("{answer}", str(a))
        completion = query_gpt(prompt_temp)
        completions.append(completion)
        try:
            score = float(completion[completion.index("[[")+2])
        except:
            score = 1.0
        scores.append(score)
    df = pd.DataFrame(completions)
    df.to_csv(save_file)
    return scores, np.array(scores).mean()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_file', type=str, required=True)
    parser.add_argument('--open_ai_key', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    questions = get_questions("results/all_questions.txt")

   
    answers = get_answers(args.answer_file)
    prompt = """[Question]
    {question}
    [The Start of Assistant’s Answer]
    {answer}
    [The End of Assistant’s Answer]"""

    OPEN_AI_KEY = args.open_ai_key

    client = OpenAI(api_key=OPEN_AI_KEY)

    scores, scores_avg = get_scores_for_qa_pairs(questions, answers, prompt,save_file=args.save_file)
    print(scores)
    print("LLM-as-a-Judge Score: "+str(scores_avg))


