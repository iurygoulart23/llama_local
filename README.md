# Projeto Llama-2-7b-hf

Este projeto permite utilizar o modelo de linguagem Llama-2-7b-hf da Meta para várias aplicações de processamento de linguagem natural.

## Pré-requisitos

- Python 3.8 ou superior
- Git
- Acesso à internet
- Conta no Hugging Face

## Como obter acesso ao modelo Llama-2-7b-hf

O modelo Llama-2-7b-hf é um modelo da Meta que requer permissão para uso. Siga os passos abaixo para obter acesso:

1. Acesse o [Hugging Face](https://huggingface.co/) e crie uma conta ou faça login
2. Visite a página do modelo [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
3. Clique no botão "Access repository" (Acessar repositório)
4. Você será redirecionado para um formulário de solicitação de acesso
5. Preencha o formulário com as informações solicitadas
6. Aguarde a aprovação da Meta (geralmente ocorre em poucos dias)
7. Uma vez aprovado, você receberá um e-mail de confirmação e terá acesso ao modelo

## Configuração do projeto

### 1. Clone o repositório do modelo

Após receber o acesso, clone o repositório do modelo para o seu ambiente local:

```bash
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
cd Llama-2-7b-hf
```

### 2. Clone este projeto

Em outro diretório, clone este projeto:

```bash
git clone [URL-DO-SEU-REPOSITORIO]
cd [NOME-DO-SEU-REPOSITORIO]
```

### 3. Execute o script de instalação

Execute o script de instalação fornecido para configurar o ambiente virtual e instalar as dependências necessárias:

```bash
chmod +x installation.sh
./installation.sh
```

O script irá:
- Criar um ambiente virtual Python
- Instalar todas as dependências necessárias listadas no arquivo `requirements.txt`
- Verificar se tudo está configurado corretamente
- Oferecer a opção de executar o `main.py` imediatamente

## Utilização

Após a instalação, você pode usar o modelo de duas maneiras:

### Opção 1: Executar o script principal

```bash
python main.py
```

### Opção 2: Importar em seu próprio código

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Caminho para o modelo baixado
model_path = "./Llama-2-7b-hf"

# Carregar tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Exemplo de uso
input_text = "Olá, como você está hoje?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Requisitos de hardware

O modelo Llama-2-7b-hf é relativamente grande e requer:
- Pelo menos 16GB de RAM
- GPU com pelo menos 8GB de VRAM (recomendado para inferência rápida)
- Aproximadamente 15GB de espaço em disco para armazenar o modelo

## Solução de problemas

### Erro de permissão ao clonar o repositório
Se você encontrar erros de permissão ao tentar clonar o repositório do Hugging Face, verifique se:
1. Sua solicitação de acesso foi aprovada
2. Você está logado no Hugging Face (via `huggingface-cli login`)

### Erro de memória ao carregar o modelo
Se você encontrar erros de memória ao carregar o modelo, tente:
1. Carregar o modelo com precisão reduzida: `model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)`
2. Usar carregamento em dispositivo específico: `device_map="auto"`
3. Caso mesmo assim tenha problemas para rodar o modelo, pode tentar rodar a versão do codigo otimizada: `main_low_vram.py`

Porém irá precisar instalar algumas novas bibliotecas:
```bash
pip install accelerate bitsandbytes
```

## Licença

Este projeto está licenciado sob GNU v3.0, mas observe que o uso do modelo Llama-2 está sujeito aos termos da Meta.

## Contribuições

Contribuições são bem-vindas!