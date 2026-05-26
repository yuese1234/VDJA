from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
import os
import json
import argparse
def moderate(chat,tokenizer,model,device):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def llama_guard(args):
    model_path = args.model_path
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    file_path = args.dataset_path
    save_path = args.save_path
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, item in enumerate(data):
        user_prompt=item["system_prompt"]+' '+ item["user_prompt"]
        answer=item["answer"]
        chat=[
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer}]
        result=moderate(chat=chat,tokenizer=tokenizer,model=model,device=device)
        parts = result.splitlines()
        item["llama_safe"] = parts[0]
        item["llama_type"] = parts[1] if len(parts) > 1 else None

    with open(save_path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model_path')
    parser.add_argument('--dataset_path', required=True, help='your_file_path')
    parser.add_argument('--save_path', required=True, help='your_save_path')
    args = parser.parse_args()
    llama_guard(args)

if __name__ == "__main__":
    main()