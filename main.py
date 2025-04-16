from transformers import pipeline
import torch

model_path = "/home/iurygoulart/projetos/llama_local/Llama-2-7b-hf"

pipe = pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Me fale um pouco sobre o SENAI-SP e sua importancia para a sociedade?"

output = pipe(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

print(output[0]["generated_text"])
