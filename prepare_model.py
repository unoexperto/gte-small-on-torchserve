import torch
from transformers import AutoTokenizer, AutoModel

# new_alloc = torch.cuda.memory.CUDAPluggableAllocator('./../torch-apu-helper/alloc.so', 'gtt_alloc', 'gtt_free')
# torch.cuda.memory.change_current_allocator(new_alloc)

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
model = AutoModel.from_pretrained("thenlper/gte-small")

tokenizer.save_pretrained('./my_tokenizer')
model.save_pretrained('./my_model')
