# AffinePNN em $\kappa_\lambda$ — Busca de Di-Higgs $\mathrm{HH} \to b\bar{b}\tau^+\tau^-$ (ggF) no CMS Run-3

Projeto modular de Mestrado (UERJ / CMS) que implementa uma **Rede Neural Parametrizada Afim (`AffinePNN`)** condicionada no acoplamento de auto-interação trilinear do bóson de Higgs ($\kappa_\lambda$) e a compara de forma estritamente controlada contra o classificador **Single-NN (`DNN_AN`) da nota de análise CMS AN-25-103 (Seção 5.3)**.

## Motivação Física e Metodológica

1. **Dependência cinemática forte em $\kappa_\lambda$ (AN-25-103, Seção 5.5.3):** Na produção não-ressonante via fusão de glúons (ggF), a interferência quântica destrutiva entre os diagramas de caixa (*box*) e triângulo altera drasticamente o espectro de massa invariante $m_{HH}$ — tornando-o duro próximo a $\kappa_\lambda = 2.45$ e muito mais suave (*low-$m_{HH}$*) em $\kappa_\lambda = 5.0$. Uma rede única não-parametrizada (`DNN_AN`) treinada misturando todas as hipóteses perde eficiência significativamente em $\kappa_\lambda = 5.0$.
2. **Arquitetura `AffinePNN` (Anzalone, Diotalevi & Bonacorsi, [arXiv:2202.00424](https://arxiv.org/abs/2202.00424)):** Em vez de apenas concatenar $\kappa_\lambda$ na entrada (`PNN_Concat`), cada bloco oculto aplica uma transformação afim condicional pós-ativação:
   $$\mathbf{z}^{(l)} = \mathbf{h}^{(l)} \odot \mathbf{s}_l(\kappa_\lambda) + \mathbf{b}_l(\kappa_\lambda)$$
   modulando escala e deslocamento de todos os mapas de representação intermediários em função da hipótese $\kappa_\lambda$ testada.
3. **Comparação 100% controlada com a AN-25-103:** Ambos os modelos (`AffinePNN` e `DNN_AN`) compartilham exatamente o mesmo *backbone* (**LBN** $N=5 \to M=10$ com zeragem física exata $(0,0,0,0)$ dos 4-vetores de objetos ausentes `bjet2`, `fatjet` e `bjet1` em eventos `0-jet`, preservando `bjet1` válido em eventos `1-jet` + **Entity Embeddings** $\text{dim}=5$ + **DenseNet** de 8 blocos `Dense(128) -> BN -> ELU`), o mesmo gerador multinomial de quotas por $(\text{subprocesso} \times \text{era})$ sem pesos negativos na loss (AN 5.3.5), o mesmo otimizador **AdamW** com *weight decay* $500 / N_{\text{pesos lineares}}$ apenas nos kernels lineares do backbone (AN 5.3.4) e o mesmo monitoramento de *early stopping* ponderado pela frequência de *oversampling* (AN 5.3.6, Fig. 50).
4. **Tratamento rigoroso dos pesos NLO na avaliação:**
   - **Classificação e histogramas de discriminação:** `ROC`, `AUC_w`, `PR-AUC`, `Precision/Recall/F1`, *working points*, *bootstrap* e histogramas padrão de score utilizam a convenção unificada $w_{\text{clf}} = \max(w_{\text{phys}}, 0)$ por evento (garantindo curvas TPR/FPR monotonicamente não-decrescentes, desduplicação da primeira ocorrência de cada `TPR` antes de interpolações e reporte simultâneo da `AUC` sem pesos).
   - **Rendimentos físicos esperados:** preservam integralmente os pesos NLO com sinal ($S = \sum_{i \in S} w_i$, $B = \sum_{i \in B} w_i$), reportando tanto a eficiência não-negativa $\epsilon_S^+$ quanto a eficiência física assinada $\epsilon_S^{\text{signed}}$, as incertezas estatísticas de Monte Carlo $\sigma_{\text{MC}} = \sqrt{\sum_i w_i^2}$ e a significância de Asimov **nominal** $Z_A$ (sem parâmetros de *nuisance*, sinalizando visualmente pontos indefinidos quando $S \le 0$ ou $B \le 0$).

---

## Estrutura do Projeto

```text
├── configs/
│   ├── default.yaml                # Manifesto principal que agrega as configurações
│   ├── features.yaml               # 49 features contínuas + 7 categóricas (Tabela 29 da AN-25-103)
│   ├── affine.yaml                 # Arquitetura AffinePNN (LBN + DenseNet + AffineConditioning)
│   ├── dnn_an.yaml                 # Baseline DNN da AN-25-103 (Single-NN sem κ_λ)
│   ├── pnn_concat.yaml             # Baseline PNN clássica (κ_λ concatenado na entrada)
│   └── training.yaml               # Caminhos, eras, quotas (AN 5.3.5), treino (AN 5.3.6) e avaliação
├── condor/
│   ├── README_condor.md            # Guia passo-a-passo de submissão no CERN lxplus + EOS + GPU
│   ├── run_job.sh                  # Wrapper bash para os worker nodes do HTCondor
│   ├── train_both.sub              # Submete treino completo (AffinePNN + DNN_AN) em 1 GPU
│   ├── train_single.sub            # Submete cada arquitetura em GPUs separadas em paralelo
│   └── sanity_studies.sub          # Job array paralelo para estudo estatístico de múltiplas seeds
├── data/
│   └── README.md                   # Formato dos Parquets no EOS, colunas e premissas físicas de pesos
├── notebooks/
│   ├── 01_parquet_analysis.ipynb       # Leitura dos Parquets, eras, reescala de pesos NLO, ESS e duplicatas
│   ├── 02_feature_analysis.ipynb       # Análise das 56 features, pré-processamento, LBN 4-vetores e m_HH vs κ_λ
│   ├── 03_training.ipynb               # Treinamento interativo via notebook (lendo configs/ e src/)
│   ├── 04_validation_diagnostics.ipynb # Validação profunda, Fig. 50 AN, KS, caudas CE/BN/LBN, matriz κ e sculpting
│   └── 05_results_and_plots.ipynb      # AffinePNN pura + AffinePNN vs DNN_AN (ROC/PR/F1/Scores log) + Export ONNX
├── outputs/
│   ├── checkpoints/                # Modelos (.keras), normalização (.csv), encoders (.json) e históricos
│   ├── exported_models/            # Pacotes completos (ONNX, ONNX End-to-End, SavedModel, .keras, .h5, .npz, SHA256)
│   ├── logs/                       # Logs de execução e logs .out/.err/.log do HTCondor
│   ├── metrics/                    # Tabelas CSV de métricas, comparações, rendimentos e diagnósticos
│   ├── plots/                      # Figuras geradas em alta resolução (affine_pure/, comparison/, performance/)
│   └── predictions/                # Arquivos Parquet de scores inferidos
├── scripts/
│   ├── train.py                    # CLI de treinamento (local, lxplus ou HTCondor) + exportação automática
│   ├── evaluate.py                 # CLI de avaliação, diagnósticos e geração completa de plots
│   ├── export_model.py             # CLI para empacotar checkpoints existentes em ONNX/SavedModel/NPZ
│   ├── inference.py                # CLI de inferência (suporta Keras ou --use-onnx via ONNX Runtime)
│   ├── sanity_studies.py           # CLI de testes pesados (random-label, interpolação e seeds)
│   └── generate_synthetic_data.py  # Gerador de Parquets sintéticos para testes fora do EOS
└── src/
    ├── data/                       # parquet_dataset.py, preprocessing.py, weights.py, sampler.py, synthetic.py
    ├── models/                     # affine_layers.py, embeddings.py, affine_dnn.py, export.py
    ├── training/                   # callbacks.py, losses.py, diagnostics.py, train.py
    ├── evaluation/                 # roc.py, significance.py, score_distributions.py, kappa_scan.py, evaluate.py, plotting.py, performance_plots.py
    └── utils/                      # config.py, logging_utils.py, paths.py, seed.py
```

---

## Instalação Rápida

```bash
pip install -r requirements.txt
pip install -e .
```

No **CERN `lxplus` / SWAN**, você também pode usar diretamente a pilha CUDA do CVMFS (`LCG_106_cuda`) conforme detalhado em [`condor/README_condor.md`](condor/README_condor.md).

---

## Como Executar

### 1. Via CERN HTCondor (Treino em lote em GPU)

```bash
chmod +x condor/run_job.sh
mkdir -p outputs/logs
condor_submit condor/train_both.sub
```

Consulte [`condor/README_condor.md`](condor/README_condor.md) para instruções completas de pré-checagem, paralelização por modelo/seed e leitura no EOS.

### 2. Via Linha de Comando (Terminal / `lxplus-gpu` / Smoke-Test Local)

```bash
# Treinar AffinePNN e DNN_AN usando a configuração padrão:
python scripts/train.py --config configs/default.yaml

# Avaliar os checkpoints salvos na validação com diagnósticos e varredura em κ_λ:
python scripts/evaluate.py --config configs/default.yaml --diagnostics --kappa-scan

# Avaliação FINAL congelada no conjunto de teste (uma única vez!):
python scripts/evaluate.py --split test --unlock-test

# Smoke-test local completo (sem precisar do EOS):
python scripts/train.py --synthetic --epochs 2 --steps-per-epoch 10
```

### 3. Via Jupyter Notebooks (SWAN ou Local)

Os 5 notebooks em `notebooks/` são modulares e importam todo o código de `src/` e `configs/`:
- Você pode analisar os dados em **`01_parquet_analysis.ipynb`** e **`02_feature_analysis.ipynb`**.
- Treinar pelo **Condor** *ou* diretamente pelo **`03_training.ipynb`** (ambos salvam os mesmos artefatos em `outputs/checkpoints/`).
- Validar a saúde da rede em **`04_validation_diagnostics.ipynb`** e gerar todas as tabelas e figuras finais em **`05_results_and_plots.ipynb`**.
