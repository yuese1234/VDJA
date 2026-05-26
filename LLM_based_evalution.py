from google import genai
from google.genai import types
import json
import PIL.Image
import pandas as pd
import time
import os
import argparse
from templates import judge_prompt
def call_api(client, model, contents, system_prompt):
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            temperature=0,
        system_instruction=system_prompt))
    return response

def eval(args):
    dataset_path=args.dataset_path
    save_path=args.save_path
    api_key=args.api_key
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, item in enumerate(data):
        system_prompt=judge_prompt.format(item["Goal"])
        user_prompt = "[PROMPT]:{}\n[ASSISTANT'S RESPONSE]: {}".format(item["system_prompt"] + item["user_prompt"],item["answer"])
        client = genai.Client(api_key=api_key)
        response =call_api(client=client,
            model="gemini-2.0-flash",
            contents=[user_prompt],
            system_prompt=system_prompt)
        item["score"] = response.text

    with open(save_path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='your_file_path')
    parser.add_argument('--save_path', required=True, help='your_save_path')
    parser.add_argument('--api_key', default='your_api_key')
    args = parser.parse_args()
    eval(args)


if __name__ == "__main__":
    main()