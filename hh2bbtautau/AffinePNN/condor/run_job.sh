#!/bin/bash
# ==============================================================================
# Wrapper de execução para Worker Nodes do CERN HTCondor (lxplus / AlmaLinux 9)
# Uso: run_job.sh <diretorio_do_projeto> <script_python> [argumentos...]
# ==============================================================================
set -euo pipefail

PROJECT_DIR="${1:?Informe o diretorio raiz do projeto como 1o argumento}"
SCRIPT_REL="${2:?Informe o script python (ex: scripts/train.py) como 2o argumento}"
shift 2

echo "=================================================================="
echo "[Condor Job] Host: $(hostname)"
echo "[Condor Job] Data: $(date)"
echo "[Condor Job] Diretorio do projeto: ${PROJECT_DIR}"
echo "[Condor Job] Script: ${SCRIPT_REL} $*"
echo "=================================================================="

# 1. Ativação do ambiente Python + CUDA
# Prioridade: (a) .venv próprio no projeto/AFS/EOS -> (b) Conda -> (c) LCG CUDA View no CVMFS
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    echo "[Env] Ativando virtualenv ${PROJECT_DIR}/.venv ..."
    source "${PROJECT_DIR}/.venv/bin/activate"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/bin/activate" ]; then
    echo "[Env] Ativando Conda ${CONDA_PREFIX} ..."
    source "${CONDA_PREFIX}/bin/activate"
elif [ -f "/cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh" ]; then
    echo "[Env] Carregando CVMFS LCG_106_cuda (AlmaLinux 9) ..."
    set +u
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh
    set -u
fi

# 2. Variáveis de ambiente recomendadas para TensorFlow / XLA em GPU no Condor
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if command -v nvidia-smi &> /dev/null; then
    echo "[GPU] Placas visíveis pelo job:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
else
    echo "[AVISO] nvidia-smi não encontrado no nó de execução!"
fi

cd "${PROJECT_DIR}"
python3 "${SCRIPT_REL}" "$@"

echo "=================================================================="
echo "[Condor Job] Finalizado em: $(date)"
echo "=================================================================="
