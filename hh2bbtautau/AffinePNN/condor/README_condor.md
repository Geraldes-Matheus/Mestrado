# Guia de Execução no CERN HTCondor (`lxplus` + `EOS` + GPU)

Este diretório contém os scripts prontos para submeter o treinamento e os estudos de validação da **AffinePNN vs DNN da AN-25-103** nos nós com GPU do cluster HTCondor do CERN.

---

## 1. Preparação inicial no `lxplus`

Conecte-se ao `lxplus` (AlmaLinux 9):

```bash
ssh <seu_usuario>@lxplus.cern.ch
```

> **Importante sobre AFS vs EOS no HTCondor:**
> Clone o código do projeto em uma área visível pelos worker nodes (por exemplo `/afs/cern.ch/work/m/mgeralde/hh_affinepnn` ou `/eos/user/m/mgeralde/...`) e mantenha os dados pesados (`parquet_dir`) e os checkpoints (`output_dir`) no **EOS**.

Torne o wrapper executável e garanta que a pasta local de logs exista:

```bash
cd /caminho/para/mestrado/hh2bbtautau/AffinePNN
chmod +x condor/run_job.sh
mkdir -p outputs/logs
```

### Ambiente Python (2 opções)

1. **Opção A — Automática via CVMFS (`LCG_106_cuda`)**:
   Se não houver `.venv`, o `condor/run_job.sh` carrega automaticamente a pilha `LCG_106_cuda` (que já traz Python, TensorFlow com CUDA, NumPy, Pandas, PyArrow, SciPy, Scikit-Learn, Matplotlib e PyYAML).
2. **Opção B — Virtualenv dedicado (`.venv`)**:
   Caso queira fixar exatamente TensorFlow/Keras >= 2.16:
   ```bash
   source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   pip install -e .
   ```

---

## 2. Teste rápido antes de submeter para a fila (Pre-Flight Check)

Antes de enviar um job longo para o Condor, valide em 20 segundos que caminhos, credenciais EOS e imports funcionam:

```bash
./condor/run_job.sh "$PWD" scripts/train.py --synthetic --epochs 1 --steps-per-epoch 5
```

Ou teste lendo uma amostra pequena real do EOS em `lxplus-gpu.cern.ch`:

```bash
./condor/run_job.sh "$PWD" scripts/train.py \
  --models affine \
  --epochs 1 \
  --steps-per-epoch 10 \
  --set data.max_events_per_file=20000
```

---

## 3. Submetendo Jobs ao HTCondor

Ajuste se necessário `EOS_PARQUETS` e `EOS_OUTPUTS` no topo dos arquivos `.sub` e renove seu ticket Kerberos:

```bash
kinit
```

### A) Treinar `AffinePNN` + `DNN_AN` no mesmo job (Recomendado)
Carrega os ~9.35M eventos uma única vez na RAM e treina os dois modelos em sequência (~1h15m em Tesla T4), salvando modelos, históricos e métricas de validação no EOS:

```bash
condor_submit condor/train_both.sub
```

### B) Treinar `AffinePNN`, `DNN_AN` e `PNN_Concat` em GPUs paralelas
Submete 3 jobs independentes em paralelo:

```bash
condor_submit condor/train_single.sub
```

### C) Rodar estudo estatístico de múltiplas seeds (Seção 13.6) em paralelo
Submete 5 jobs (seeds `101..105`) simultaneamente em 5 GPUs:

```bash
condor_submit condor/sanity_studies.sub
```

---

## 4. Monitoramento dos Jobs

```bash
# Ver fila de jobs ativos/pendentes
condor_q

# Acompanhar saída do treino em tempo real (sem barras ANSI quebradas: o código detecta batch automaticamente!)
condor_tail -f <ClusterId>.0

# Em caso de job em Hold (ex: credencial expirada)
condor_q -better-analyze <ClusterId>
```

---

## 5. Analisando os Resultados em Notebooks (SWAN ou Local)

Assim que o job do Condor terminar, os seguintes arquivos estarão prontos em `$(EOS_OUTPUTS)/checkpoints/`:
- `model_AffinePNN.keras` e `model_DNN_AN.keras`
- `history_AffinePNN.json` / `.csv` e `history_DNN_AN.json` / `.csv`
- `normalization_train_only.csv` e `config_encoders.json`
- `run_config_snapshot.json`

Abra diretamente os notebooks `04_validation_diagnostics.ipynb` e `05_results_and_plots.ipynb` no **CERN SWAN** (ou localmente) apontando `OUTPUT_DIR` para `$(EOS_OUTPUTS)`: eles carregarão os modelos prontos sem precisar treinar novamente!
