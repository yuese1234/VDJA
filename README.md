# Safety Guardrails of Large Language Models Are Vulnerable to Value-Driven Adversarial Prompting


## Attack on target LLMs
```python
python attack.py
--target_LLM 'target_LLM' \
--dataset-path 'your_dataset_path' \
--save-path 'your_save_path' \
```

## LLM_based evalution
```python
python LLM_based_evalution.py 
    --result-path 'attack_result_path' \
    --save-path 'save_dir' \
```
## Perplexity Filter defense
```python
python perplexity_filter.py 
    --model-path 'your_model_path' \
    --dataset-path  'your_dataset_dir' \
    --save-path 'save_dir' \
```
## Llama Guard 
```python
python llama_guard_defense.py 
    --model-path 'your_model_path' \
    --dataset-path  'your_dataset_dir' \
    --save-path 'save_dir' \
```
## Self-Reminder defense
```python
python Self_reminder_defense.py
--target_LLM 'target_LLM' \
--dataset-path 'your_dataset_path' \
--save-path 'your_save_path' \
```


