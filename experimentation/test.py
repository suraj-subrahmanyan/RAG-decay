from pydantic import BaseModel
import outlines
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Response(BaseModel):
    reasoning: str
    answer: int


prompt = '''
 You are a helpful assistant that can answer questions about math problems.

 Question: {question}
 Output:
{{
    answer: int
}}
the answer should be a single integer
'''
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Thinking-2507", 
    dtype=torch.float16,
attn_implementation="flash_attention_2").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")

resp = outlines.from_transformers(model, tokenizer)

user_query = prompt.format(question="what is 3 + 5 * 4")
print(resp("what is 1+1", int, max_new_tokens=1024, temperature=0.1))