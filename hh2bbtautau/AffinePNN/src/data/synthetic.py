"""Gerador de arquivos Parquet sintéticos fisicamente consistentes para smoke-tests locais."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _make_four_vector(rng: np.random.Generator, n: int, pt_scale: float, mass: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pt = rng.gamma(shape=2.5, scale=pt_scale, size=n).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    eta = rng.normal(0.0, 1.2, size=n).astype(np.float32)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    m_smear = np.maximum(rng.normal(mass, max(mass * 0.08, 0.5), size=n), 0.1).astype(np.float32)
    e = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m_smear ** 2).astype(np.float32)
    return e, px, py, pz


def generate_synthetic_parquets(
    output_dir: str | Path,
    cfg: dict[str, Any],
    events_per_bkg_file: int = 8_000,
    events_per_sig_file: int = 2_500,
    seed: int = 42,
) -> Path:
    """Cria `{era}_{group}.parquet` sintéticos respeitando 4-vetores relativísticos (E^2 >= |p|^2),
    domínios da Tabela 29, pesos negativos NLO no DY e dependência cinemática real em `kappa_lambda`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    eras = cfg["data"]["eras"]
    groups = cfg["data"]["groups"]
    kappa_map = cfg["physics"]["kappa_map"]
    weight_col = cfg["features"]["weight_col"]

    dy_samples = [
        "dy_tautau_m50toinf_2j_filtered_amcatnlo",
        "dy_m50toinf_2j_amcatnlo",
        "dy_tautau_m50toinf_1j_amcatnlo",
        "dy_m50toinf_0j_amcatnlo",
    ]
    tt_samples = ["tt_dl_powheg", "tt_sl_powheg", "tt_fh_powheg"]

    for era in eras:
        for group in groups:
            is_sig = group in kappa_map
            n = int(events_per_sig_file if is_sig else events_per_bkg_file)
            label = 2 if is_sig else (0 if group == "dy" else 1)
            kl_val = float(kappa_map[group]) if is_sig else 0.0

            # Cinemática dependente da classe e de kappa_lambda (kl=5 mais soft em m_HH; kl=2.45 mais duro)
            if label == 0:  # DY (pico em m_Z ~ 91 GeV)
                tau_scale, b_scale, m_tt_base, m_bb_base, btag_mu = 28.0, 22.0, 91.0, 85.0, 0.25
            elif label == 1:  # ttbar
                tau_scale, b_scale, m_tt_base, m_bb_base, btag_mu = 38.0, 48.0, 110.0, 135.0, 0.72
            else:  # HH sinal parametrizado em kappa_lambda
                boost_factor = {0.0: 1.05, 1.0: 1.25, 2.45: 1.50, 5.0: 0.78}.get(kl_val, 1.1)
                tau_scale = 42.0 * boost_factor
                b_scale = 45.0 * boost_factor
                m_tt_base, m_bb_base, btag_mu = 125.0, 125.0, 0.85

            has_jet_pair = rng.choice([0, 1], size=n, p=[0.08, 0.92]).astype(np.int32)
            has_fatjet = rng.choice([0, 1], size=n, p=[0.92, 0.08]).astype(np.int32)

            t1_e, t1_px, t1_py, t1_pz = _make_four_vector(rng, n, tau_scale, 1.777)
            t2_e, t2_px, t2_py, t2_pz = _make_four_vector(rng, n, tau_scale * 0.8, 1.777)
            b1_e, b1_px, b1_py, b1_pz = _make_four_vector(rng, n, b_scale, 4.18)
            b2_e, b2_px, b2_py, b2_pz = _make_four_vector(rng, n, b_scale * 0.75, 4.18)
            fj_e, fj_px, fj_py, fj_pz = _make_four_vector(rng, n, b_scale * 2.2, 125.0)

            # Zera objetos ausentes exatamente como nos parquets reais
            for arr in (b2_e, b2_px, b2_py, b2_pz):
                arr[has_jet_pair == 0] = 0.0
            for arr in (fj_e, fj_px, fj_py, fj_pz):
                arr[has_fatjet == 0] = 0.0

            htt_px, htt_py, htt_pz = t1_px + t2_px, t1_py + t2_py, t1_pz + t2_pz
            htt_e = np.sqrt(htt_px ** 2 + htt_py ** 2 + htt_pz ** 2 + m_tt_base ** 2).astype(np.float32)

            hbb_px, hbb_py, hbb_pz = b1_px + b2_px, b1_py + b2_py, b1_pz + b2_pz
            hbb_e = np.sqrt(hbb_px ** 2 + hbb_py ** 2 + hbb_pz ** 2 + m_bb_base ** 2).astype(np.float32)

            hh_px, hh_py, hh_pz = htt_px + hbb_px, htt_py + hbb_py, htt_pz + hbb_pz
            hh_e = (htt_e + hbb_e).astype(np.float32)
            for arr in (hbb_e, hbb_px, hbb_py, hbb_pz, hh_e, hh_px, hh_py, hh_pz):
                arr[has_jet_pair == 0] = 0.0

            hfj_px, hfj_py, hfj_pz = htt_px + fj_px, htt_py + fj_py, htt_pz + fj_pz
            hfj_e = (htt_e + fj_e).astype(np.float32)
            for arr in (hfj_e, hfj_px, hfj_py, hfj_pz):
                arr[has_fatjet == 0] = 0.0

            # Pesos físicos com fração realista de pesos negativos NLO (~30% em DY, ~5% em HH, ~0.4% em ttbar)
            if label == 0:
                signs = rng.choice([1.0, -1.0], size=n, p=[0.70, 0.30])
                w_norm = (signs * rng.lognormal(mean=-1.2, sigma=0.5, size=n)).astype(np.float32)
                samples = rng.choice(dy_samples, size=n, p=[0.45, 0.30, 0.20, 0.05])
            elif label == 1:
                signs = rng.choice([1.0, -1.0], size=n, p=[0.996, 0.004])
                w_norm = (signs * rng.lognormal(mean=-1.5, sigma=0.2, size=n)).astype(np.float32)
                samples = rng.choice(tt_samples, size=n, p=[0.78, 0.21, 0.01])
            else:
                signs = rng.choice([1.0, -1.0], size=n, p=[0.94, 0.06])
                w_norm = (signs * rng.lognormal(mean=-8.5, sigma=0.25, size=n)).astype(np.float32)
                samples = np.full(n, f"{group}_kt1_powheg")

            b2_present = (has_jet_pair == 1).astype(np.float32)
            data = {
                "reg_dnn_moe_met_px": rng.normal(0.0, 35.0, size=n).astype(np.float32),
                "reg_dnn_moe_met_py": rng.normal(0.0, 35.0, size=n).astype(np.float32),
                "reg_dnn_moe_met_cov00": rng.lognormal(5.5, 0.9, size=n).astype(np.float32),
                "reg_dnn_moe_met_cov01": rng.normal(0.0, 80.0, size=n).astype(np.float32),
                "reg_dnn_moe_met_cov11": rng.lognormal(5.5, 0.9, size=n).astype(np.float32),
                "reg_dnn_moe_vis_tau1_e": t1_e,
                "reg_dnn_moe_vis_tau1_px": t1_px,
                "reg_dnn_moe_vis_tau1_py": t1_py,
                "reg_dnn_moe_vis_tau1_pz": t1_pz,
                "reg_dnn_moe_vis_tau2_e": t2_e,
                "reg_dnn_moe_vis_tau2_px": t2_px,
                "reg_dnn_moe_vis_tau2_py": t2_py,
                "reg_dnn_moe_vis_tau2_pz": t2_pz,
                "reg_dnn_moe_bjet1_e": b1_e,
                "reg_dnn_moe_bjet1_px": b1_px,
                "reg_dnn_moe_bjet1_py": b1_py,
                "reg_dnn_moe_bjet1_pz": b1_pz,
                "reg_dnn_moe_bjet1_tag_b": np.clip(rng.normal(btag_mu, 0.18, size=n), 0.0, 1.0).astype(np.float32),
                "reg_dnn_moe_bjet1_tag_cvsb": np.clip(rng.uniform(0.0, 1.0, size=n), 0.0, 1.0).astype(np.float32),
                "reg_dnn_moe_bjet1_tag_cvsl": np.clip(rng.uniform(0.0, 1.0, size=n), 0.0, 1.0).astype(np.float32),
                "reg_dnn_moe_bjet1_hhbtag": np.clip(rng.normal(btag_mu, 0.15, size=n), 0.0, 1.0).astype(np.float32),
                "reg_dnn_moe_bjet2_e": b2_e,
                "reg_dnn_moe_bjet2_px": b2_px,
                "reg_dnn_moe_bjet2_py": b2_py,
                "reg_dnn_moe_bjet2_pz": b2_pz,
                "reg_dnn_moe_bjet2_tag_b": (np.clip(rng.normal(btag_mu * 0.9, 0.2, size=n), 0.0, 1.0) * b2_present).astype(np.float32),
                "reg_dnn_moe_bjet2_tag_cvsb": (rng.uniform(0.0, 1.0, size=n) * b2_present).astype(np.float32),
                "reg_dnn_moe_bjet2_tag_cvsl": (rng.uniform(0.0, 1.0, size=n) * b2_present).astype(np.float32),
                "reg_dnn_moe_bjet2_hhbtag": (np.clip(rng.normal(btag_mu * 0.9, 0.18, size=n), 0.0, 1.0) * b2_present).astype(np.float32),
                "reg_dnn_moe_fatjet_e": fj_e,
                "reg_dnn_moe_fatjet_px": fj_px,
                "reg_dnn_moe_fatjet_py": fj_py,
                "reg_dnn_moe_fatjet_pz": fj_pz,
                "reg_dnn_moe_htt_e": htt_e,
                "reg_dnn_moe_htt_px": htt_px,
                "reg_dnn_moe_htt_py": htt_py,
                "reg_dnn_moe_htt_pz": htt_pz,
                "reg_dnn_moe_hbb_e": hbb_e,
                "reg_dnn_moe_hbb_px": hbb_px,
                "reg_dnn_moe_hbb_py": hbb_py,
                "reg_dnn_moe_hbb_pz": hbb_pz,
                "reg_dnn_moe_htthbb_e": hh_e,
                "reg_dnn_moe_htthbb_px": hh_px,
                "reg_dnn_moe_htthbb_py": hh_py,
                "reg_dnn_moe_htthbb_pz": hh_pz,
                "reg_dnn_moe_httfatjet_e": hfj_e,
                "reg_dnn_moe_httfatjet_px": hfj_px,
                "reg_dnn_moe_httfatjet_py": hfj_py,
                "reg_dnn_moe_httfatjet_pz": hfj_pz,
                "reg_dnn_moe_pair_type": rng.choice([0, 1, 2], size=n).astype(np.int32),
                "reg_dnn_moe_dm1": rng.choice([-1, 0, 1, 10, 11], size=n).astype(np.int32),
                "reg_dnn_moe_dm2": rng.choice([0, 1, 10, 11], size=n).astype(np.int32),
                "reg_dnn_moe_vis_tau1_charge": rng.choice([-1, 1], size=n).astype(np.int32),
                "reg_dnn_moe_vis_tau2_charge": rng.choice([-1, 1], size=n).astype(np.int32),
                "reg_dnn_moe_has_jet_pair": has_jet_pair,
                "reg_dnn_moe_has_fatjet": has_fatjet,
                "kappa_lambda": np.full(n, kl_val if is_sig else np.nan, dtype=np.float32),
                weight_col: w_norm,
                "label": np.full(n, label, dtype=np.int32),
                "sample": samples,
            }

            df = pd.DataFrame(data)
            df.to_parquet(out / f"{era}_{group}.parquet", index=False)

    return out
