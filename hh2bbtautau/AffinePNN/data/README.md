# Dados de Entrada (`Parquet`) — CMS Run-3 $\mathrm{HH} \to b\bar{b}\tau^+\tau^-$ (ggF)

Os arquivos Parquet pré-processados residem por padrão no **CERN EOS**:

```text
/eos/user/m/mgeralde/Analysis_hh2bbtautau/PNN/processed_data/parquet/
```

## Convenção de Nomes (`{era}_{group}.parquet`)

- **Eras do Run-3 consideradas (`data.eras`)**:
  - `22pre` (`2022preEE`)
  - `22post` (`2022postEE`)
  - `23pre` (`2023preBPix`)
  - `23post` (`2023postBPix`)
- **Grupos de processos (`data.groups`)**:
  - Fundos principais: `dy` ($\mathrm{Z/\gamma^* \to \ell\ell/\tau\tau + jets}$, NLO aMC@NLO) e `tt` ($t\bar{t}$ Powheg: DL, SL, FH)
  - Sinal ggF $\mathrm{HH} \to b\bar{b}\tau\tau$ parametrizado em $\kappa_\lambda$:
    - `hh_ggf_kl0` ($\kappa_\lambda = 0.0$)
    - `hh_ggf_kl1` ($\kappa_\lambda = 1.0$, Modelo Padrão)
    - `hh_ggf_kl2p45` ($\kappa_\lambda = 2.45$, interferência destrutiva máxima)
    - `hh_ggf_kl5` ($\kappa_\lambda = 5.0$)

## Colunas Esperadas (60 colunas)

Definidas em [`configs/features.yaml`](../configs/features.yaml) em conformidade com a **Tabela 29 da AN-25-103 (Seção 5.3.2)**:
- **49 features contínuas** (`reg_dnn_moe_*`): MET ($p_x, p_y$ e matriz de covariância), 4-vetores $(E, p_x, p_y, p_z)$ rotacionados em $\phi$ de $\tau_1, \tau_2, b_1, b_2$, AK8 `fatjet`, candidatos reconstruídos $H_{\tau\tau}, H_{bb}, HH_{bb\tau\tau}, H_{\tau\tau}+\text{FatJet}$, além dos scores ParticleNet/UParT e HHbtag dos dois b-jets.
- **7 features categóricas**: `pair_type`, `dm1`, `dm2`, `vis_tau1_charge`, `vis_tau2_charge`, `has_jet_pair`, `has_fatjet`.
- **4 colunas de metadados** (*jamais passadas como feature à rede*): `kappa_lambda`, `normalization_weight`, `label` (`0=DY`, `1=ttbar`, `2=HH`) e `sample` (nome do subprocesso MC).

## Premissas Físicas sobre `normalization_weight`

1. `normalization_weight` já incorpora $\sigma \times \mathcal{L}_{\text{era}} \times \text{SFs} / \sum w_{\text{gen}}$ da amostra completa. Quando `max_events_per_file` limita a leitura de arquivos gigantes de fundo, o código aplica automaticamente o fator de reescala por arquivo:
   $$w_{\text{phys}} = \text{normalization\_weight} \times \frac{N_{\text{total}}}{N_{\text{lidos}}}$$
2. Os sinais negativos provenientes da geração MC em NLO (especialmente ~32% no Drell-Yan aMC@NLO) são físicos e **preservados** no cálculo das quotas líquidas por subprocesso, nas curvas ROC ponderadas e nos rendimentos esperados.

## Dados Sintéticos para Testes Locais

Quando estiver fora do `lxplus`/EOS, gere um conjunto sintético fisicamente consistente em `data/synthetic/` com:

```bash
python scripts/generate_synthetic_data.py --output-dir data/synthetic
```
