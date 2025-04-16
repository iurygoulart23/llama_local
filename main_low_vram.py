from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Configurando a quantização 4-bit com suporte a offload para CPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True  # Habilita offload para CPU
)

# Caminho para o modelo
model_path = "/home/iurygoulart/projetos/llama_local/Llama-2-7b-hf"

# Carregar tokenizer
print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configurar device_map personalizado para controlar exatamente quais módulos vão para CPU vs GPU
# Esta é uma abordagem mais conservadora que coloca apenas algumas camadas na GPU
print("Carregando modelo com quantização 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="balanced_low_0",  # Estratégia de balanceamento conservadora
    max_memory={0: "3GiB", "cpu": "12GiB"},  # Limita memória na GPU e reserva RAM
    offload_folder="offload",  # Pasta para offload
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

# Prompt
prompt = "Me fale um pouco sobre o SENAI-SP e sua importancia para a sociedade?"
print(f"\nPrompt: {prompt}")
print("\nGerando resposta...")

# Tokenizar o prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# Gerar resposta com configurações para economizar memória
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        use_cache=True,
    )

# Decodificar e imprimir a resposta
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n=== RESPOSTA ===")
print(response)
print("================")