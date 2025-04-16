#!/bin/bash
# Define cores para melhor visualização
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Função para exibir mensagens de status
function echo_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Função para exibir mensagens de aviso
function echo_warning() {
    echo -e "${YELLOW}[AVISO]${NC} $1"
}

# Função para exibir mensagens de erro
function echo_error() {
    echo -e "${RED}[ERRO]${NC} $1"
}

# Função para verificar se o comando foi executado com sucesso
function check_status() {
    if [ $? -eq 0 ]; then
        echo_status "$1"
    else
        echo_error "$2"
        exit 1
    fi
}

# Banner do início da instalação
echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN}      CONFIGURAÇÃO DO AMBIENTE LLAMA-2-7b-hf        ${NC}"
echo -e "${GREEN}====================================================${NC}\n"

# Verificar se virtualenv está instalado
echo_status "Verificando se virtualenv está instalado..."
if ! command -v virtualenv &> /dev/null; then
    echo_warning "virtualenv não encontrado. Instalando..."
    pip install virtualenv
    check_status "virtualenv instalado com sucesso." "Falha ao instalar virtualenv."
fi

# Diretório para o ambiente virtual
PROJECT_DIR="./Llama-2-7b-hf"
VENV_DIR="$PROJECT_DIR/.venv"

# Criando o diretório do projeto se não existir
echo_status "Criando diretório do projeto em $PROJECT_DIR..."
mkdir -p "$PROJECT_DIR"
check_status "Diretório do projeto criado com sucesso." "Falha ao criar diretório do projeto."

# Verificar se o ambiente virtual já existe
if [ -d "$VENV_DIR" ]; then
    echo_warning "Ambiente virtual já existe em $VENV_DIR."
    read -p "Deseja recriar o ambiente virtual? (s/n): " recreate_venv
    if [[ $recreate_venv == "s" || $recreate_venv == "S" ]]; then
        echo_status "Removendo ambiente virtual existente..."
        rm -rf "$VENV_DIR"
        check_status "Ambiente virtual removido com sucesso." "Falha ao remover ambiente virtual."
    else
        echo_status "Usando o ambiente virtual existente."
    fi
fi

# Criando o ambiente virtual (se necessário)
if [ ! -d "$VENV_DIR" ]; then
    echo_status "Criando ambiente virtual em $VENV_DIR..."
    virtualenv "$VENV_DIR"
    check_status "Ambiente virtual criado com sucesso." "Falha ao criar ambiente virtual."
fi

# Ativando o ambiente virtual
echo_status "Ativando ambiente virtual..."
source "$VENV_DIR/bin/activate"
check_status "Ambiente virtual ativado com sucesso." "Falha ao ativar ambiente virtual."

# Verificando se o Python está funcionando corretamente
echo_status "Verificando versão do Python..."
python --version
check_status "Python verificado com sucesso." "Falha ao verificar a versão do Python."

# Atualizando pip
echo_status "Atualizando pip para a versão mais recente..."
pip install --upgrade pip
check_status "Pip atualizado com sucesso." "Falha ao atualizar pip."

# Verificando se o arquivo requirements.txt existe
if [ -f "requirements.txt" ]; then
    echo_status "Instalando dependências do arquivo requirements.txt..."
    pip install -r requirements.txt
    check_status "Dependências instaladas com sucesso!" "Falha ao instalar dependências."
else
    echo_error "Arquivo requirements.txt não encontrado no diretório atual."
    echo_warning "Por favor, certifique-se de que o arquivo existe e tente novamente."
    exit 1
fi

# Verificar espaço em disco
echo_status "Verificando espaço em disco disponível..."
df -h .
echo_status "Certifique-se de que há espaço suficiente para o modelo Llama-2-7b-hf."

# Informações finais
echo -e "\n${GREEN}====================================================${NC}"
echo_status "Ambiente virtual configurado com sucesso em $VENV_DIR"
echo_status "Para ativar manualmente o ambiente virtual, execute:"
echo -e "  ${YELLOW}source $VENV_DIR/bin/activate${NC}"
echo_status "Para executar o projeto, use:"
echo -e "  ${YELLOW}python main.py${NC}"
echo -e "${GREEN}====================================================${NC}\n"

# Perguntar se deseja executar o main.py agora
read -p "Deseja executar o main.py agora? (s/n): " run_main
if [[ $run_main == "s" || $run_main == "S" ]]; then
    if [ -f "main.py" ]; then
        echo_status "Executando main.py..."
        python main.py
    else
        echo_warning "Arquivo main.py não encontrado no diretório atual."
    fi
else
    echo_status "Instalação concluída. Execute 'python main.py' quando estiver pronto."
fi