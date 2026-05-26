from transformers import GPT2Tokenizer, GPT2Model,GPT2LMHeadModel
import gc
import os
import torch
import json
import argparse

def PPL_filter(args):
    device="cuda"
    max_length=1024
    stride=1024
    model_path=args.model_path
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()
    dataset_path=args.dataset_path
    save_path=args.save_path
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for i, item in enumerate(data):
        text = f'{item["system_prompt"]} {item["user_prompt"]}'
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        seq_len = input_ids.size(1)
    
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids_piece = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids_piece.clone()
            target_ids[:, :-trg_len] = -100
    
            with torch.no_grad():
                outputs = model(input_ids_piece, labels=target_ids)
                neg_log_likelihood = outputs.loss
    
            nlls.append(neg_log_likelihood)
    
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
    
            del input_ids_piece, outputs, neg_log_likelihood
    
        ppl = torch.exp(torch.stack(nlls).mean()).item()
        item["PPL_GPT2"]= round(ppl, 2)
        del nlls
        gc.collect()
        torch.cuda.empty_cache()
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='your_model_path')
    parser.add_argument('--dataset_path', required=True, help='your_file_path')
    parser.add_argument('--save_path', required=True, help='your_save_path')
    args = parser.parse_args()
    PPL_filter(args)
if __name__ == "__main__":
    main()
