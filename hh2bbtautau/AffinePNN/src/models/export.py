"""Exportação completa e autossuficiente de modelos para produção, dissertação e CMS (CMSSW/Columnflow):

Salva em um único diretório versionado (`outputs/exported_models/<ModelName>/`):
1. `model_<Name>.keras`              — Arquivo nativo Keras v3 (arquitetura + camadas customizadas + pesos).
2. `<Name>.weights.h5`               — Pesos puros em HDF5 (backup independente de serialização de grafo).
3. `saved_model/`                    — TensorFlow `SavedModel` nativo (C++ / TF Serving).
4. `<Name>.onnx`                     — Grafo ONNX padrão (recebe tensores já pré-processados/normalizados).
5. `<Name>_end2end_raw.onnx`         — Grafo ONNX End-to-End Físico Autossuficiente: recebe diretamente as
                                       49 features contínuas brutas (GeV/scores), as 7 categóricas brutas do CMS
                                       e `kappa_lambda` físico bruto (sem precisar de CSV/JSON externos em C++/Python!),
                                       executando mascaramento de objetos ausentes, `sign(x)*log1p(|x|)` na MET cov,
                                       z-score, mapeamento categórico e `kappa / kappa_norm` dentro do próprio grafo ONNX.
6. `learned_parameters_<Name>.npz`   — Todos os pesos aprendidos exportados como arrays NumPy nomeados
                                       (incluindo combinações de partículas/referenciais do LBN e escalas/biases afins).
7. `architecture_<Name>.json`        — Configuração completa da arquitetura e tabela detalhada por camada.
8. `preprocessing_spec.json`         — Médias, desvios-padrão, vocabulários categóricos, máscaras e `kappa_norm`,
   + `normalization_train_only.csv`    além de cópias dos CSV/JSON e YAML completo do experimento.
   + `config_encoders.json`
   + `training_config_full.yaml`
9. `bundle_manifest.json`            — Manifesto auditável com hashes SHA-256, versões de pacotes, assinaturas
                                       de tensores e relatório de paridade numérica Keras vs ONNX Runtime (`max |ΔP|`).
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data.parquet_dataset import short_cat_name
from src.data.preprocessing import CategoricalPreprocessor, FeatureScaler, make_inputs
from src.models.affine_layers import AffineConditioning, LorentzBoostNetwork


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_environment_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    }
    for pkg in ("onnx", "onnxruntime", "tf2onnx", "pyarrow", "scipy", "sklearn"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def export_learned_parameters_npz(model: keras.Model, out_path: Path) -> dict[str, Any]:
    """Salva todos os tensores de pesos (treináveis e não-treináveis) em `.npz` e retorna resumo estruturado."""
    arrays: dict[str, np.ndarray] = {}
    layer_summaries: list[dict[str, Any]] = []

    for layer in model.layers:
        w_list = layer.weights
        n_trainable = int(sum(np.prod(w.shape) for w in layer.trainable_weights))
        n_non_trainable = int(sum(np.prod(w.shape) for w in layer.non_trainable_weights))
        entry: dict[str, Any] = {
            "name": layer.name,
            "class_name": layer.__class__.__name__,
            "trainable_params": n_trainable,
            "non_trainable_params": n_non_trainable,
            "weights": [],
        }

        if isinstance(layer, AffineConditioning):
            # Subcamadas scale e bias
            for sub_name, sub_layer in (("scale", layer.scale_layer), ("bias", layer.bias_layer)):
                for w in sub_layer.weights:
                    arr = keras.ops.convert_to_numpy(w)
                    w_tag = w.path.replace("/", "__").replace(":", "_")
                    key = f"{layer.name}__{sub_name}__{w_tag}"
                    arrays[key] = arr
                    entry["weights"].append({"key": key, "shape": list(arr.shape), "dtype": str(arr.dtype)})
        else:
            for idx, w in enumerate(w_list):
                arr = keras.ops.convert_to_numpy(w)
                w_name = getattr(w, "name", f"w{idx}").split(":")[0].replace("/", "__")
                key = f"{layer.name}__{w_name}"
                if key in arrays:
                    key = f"{key}_{idx}"
                arrays[key] = arr
                entry["weights"].append({"key": key, "shape": list(arr.shape), "dtype": str(arr.dtype)})

        if isinstance(layer, LorentzBoostNetwork):
            entry["lbn_n_particles"] = layer.n_particles
            entry["lbn_n_combined"] = layer.n_combined

        layer_summaries.append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return {
        "npz_file": out_path.name,
        "num_tensors": len(arrays),
        "total_params": int(model.count_params()),
        "layers": layer_summaries,
    }


def _attach_onnx_metadata(onnx_path: Path, metadata: dict[str, str]) -> None:
    """Injeta metadados físicos e de colunas diretamente no cabeçalho protobuf do arquivo `.onnx`."""
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    existing = {p.key: p for p in onnx_model.metadata_props}
    for k, v in metadata.items():
        if k in existing:
            existing[k].value = str(v)
        else:
            prop = onnx_model.metadata_props.add()
            prop.key = str(k)
            prop.value = str(v)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(onnx_path))


def export_preprocessed_onnx(
    model: keras.Model,
    onnx_path: Path,
    n_continuous: int,
    cat_vocab_sizes: dict[str, int],
    opset: int = 17,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Exporta o modelo Keras para ONNX recebendo os tensores pré-processados (mesma assinatura do Keras)."""
    from keras.src.export.tf2onnx_lib import patch_tf2onnx
    import tf2onnx

    specs = [tf.TensorSpec((None, int(n_continuous)), tf.float32, name="x_continuous")]
    for short_name in cat_vocab_sizes:
        specs.append(tf.TensorSpec((None, 1), tf.int32, name=f"cat_{short_name}"))
    specs.append(tf.TensorSpec((None, 1), tf.float32, name="kappa_lambda"))

    @tf.function
    def forward_preprocessed(*tensors):
        probs = model(list(tensors), training=False)
        return {"class_probs": probs}

    patch_tf2onnx()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(
        forward_preprocessed,
        input_signature=specs,
        opset=int(opset),
        output_path=str(onnx_path),
    )

    if metadata:
        _attach_onnx_metadata(onnx_path, metadata)

    return {
        "file": onnx_path.name,
        "opset": int(opset),
        "inputs": [{"name": s.name, "shape": ["batch", int(s.shape[1])], "dtype": s.dtype.name} for s in specs],
        "outputs": [{"name": "class_probs", "shape": ["batch", int(model.output_shape[-1])], "dtype": "float32"}],
    }


def export_end2end_raw_onnx(
    model: keras.Model,
    onnx_path: Path,
    scaler: FeatureScaler,
    cat_preprocessor: CategoricalPreprocessor,
    cfg: dict[str, Any],
    opset: int = 17,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Constrói e exporta um grafo ONNX End-to-End que embute TODO o pré-processamento físico e estatístico.

    Entradas do grafo ONNX:
      - `raw_continuous`   : float32 `[batch, 49]` (valores físicos brutos dos Parquets/NanoAOD)
      - `raw_categorical`  : int32   `[batch, 7]`  (inteiros brutos das 7 variáveis categóricas)
      - `raw_kappa_lambda` : float32 `[batch, 1]`  (valor físico real de κ_λ, ex.: 0.0, 1.0, 2.45, 5.0)
    Saída:
      - `class_probs`      : float32 `[batch, 3]`  (`[P(DY), P(ttbar), P(HH)]`)
    """
    from keras.src.export.tf2onnx_lib import patch_tf2onnx
    import tf2onnx

    fcfg = cfg["features"]
    cont_cols: list[str] = list(scaler.features)
    cat_cols: list[str] = list(cat_preprocessor.categorical_features)
    sentinels = [float(s) for s in fcfg.get("sentinels", [-99999.0, -9999.0, -999.0])]
    cov_cols = set(fcfg.get("cov_cols", []))
    missing_masks: dict[str, str] = dict(fcfg.get("missing_object_masks", {}))
    kappa_norm = float(cfg["physics"]["kappa_norm"])

    mean_np = np.asarray(scaler.mean, dtype=np.float32).reshape(1, -1)
    std_np = np.asarray(scaler.std, dtype=np.float32).reshape(1, -1)
    cov_mask_np = np.array([[c in cov_cols for c in cont_cols]], dtype=bool)

    # Mapeamento (flag_cat_idx, mask_bool_cont[1, 49]) para objetos opcionais ausentes (bjet2/hbb/htthbb e fatjet/httfatjet)
    missing_rules: list[tuple[int, np.ndarray]] = []
    for flag_col, prefixes_cfg in missing_masks.items():
        if flag_col in cat_cols:
            cat_idx = cat_cols.index(flag_col)
            prefixes = [prefixes_cfg] if isinstance(prefixes_cfg, str) else list(prefixes_cfg)
            affected = np.array([[any(c.startswith(pfx) for pfx in prefixes) for c in cont_cols]], dtype=bool)
            if affected.any():
                missing_rules.append((cat_idx, affected))

    mask_bjet1_always = bool(fcfg.get("mask_bjet1_always_on_no_pair", False))
    bjet1_affected_np = np.array([[c.startswith("reg_dnn_moe_bjet1_") for c in cont_cols]], dtype=bool)
    bjet1_e_idx = cont_cols.index("reg_dnn_moe_bjet1_e") if "reg_dnn_moe_bjet1_e" in cont_cols else None
    has_pair_cat_idx = cat_cols.index("reg_dnn_moe_has_jet_pair") if "reg_dnn_moe_has_jet_pair" in cat_cols else None

    # Tabelas constantes de codificação para cada uma das 7 features categóricas
    cat_tables: list[tuple[list[int], list[int], int]] = []
    for c_name in cat_cols:
        enc = cat_preprocessor.encoders[c_name]
        items = sorted(enc.value_to_index.items(), key=lambda kv: kv[1])
        k_vals = [int(k) for k, _ in items]
        k_idxs = [int(v) for _, v in items]
        unk_idx = int(enc.vocab_size - 1)
        cat_tables.append((k_vals, k_idxs, unk_idx))

    specs = [
        tf.TensorSpec((None, len(cont_cols)), tf.float32, name="raw_continuous"),
        tf.TensorSpec((None, len(cat_cols)), tf.int32, name="raw_categorical"),
        tf.TensorSpec((None, 1), tf.float32, name="raw_kappa_lambda"),
    ]

    @tf.function
    def forward_end2end(raw_continuous, raw_categorical, raw_kappa_lambda):
        is_valid = tf.math.is_finite(raw_continuous)
        for sv in sentinels:
            is_valid = tf.logical_and(
                is_valid,
                tf.not_equal(raw_continuous, tf.constant(sv, dtype=tf.float32)),
            )

        for cat_idx, affected_np in missing_rules:
            flag_zero = tf.equal(raw_categorical[:, cat_idx : cat_idx + 1], 0)
            affected_tf = tf.constant(affected_np, dtype=tf.bool)
            mask_drop = tf.logical_and(flag_zero, affected_tf)
            is_valid = tf.logical_and(is_valid, tf.logical_not(mask_drop))

        if has_pair_cat_idx is not None and bjet1_affected_np.any():
            no_pair_tf = tf.equal(raw_categorical[:, has_pair_cat_idx : has_pair_cat_idx + 1], 0)
            if mask_bjet1_always:
                b1_drop = tf.logical_and(no_pair_tf, tf.constant(bjet1_affected_np, dtype=tf.bool))
                is_valid = tf.logical_and(is_valid, tf.logical_not(b1_drop))
            elif bjet1_e_idx is not None:
                b1_e_raw = raw_continuous[:, bjet1_e_idx : bjet1_e_idx + 1]
                b1_zero_tf = tf.logical_or(tf.logical_not(tf.math.is_finite(b1_e_raw)), b1_e_raw <= 0.0)
                b1_drop = tf.logical_and(
                    tf.logical_and(no_pair_tf, b1_zero_tf),
                    tf.constant(bjet1_affected_np, dtype=tf.bool),
                )
                is_valid = tf.logical_and(is_valid, tf.logical_not(b1_drop))

        x_safe = tf.where(is_valid, raw_continuous, tf.zeros_like(raw_continuous))
        abs_x = tf.math.abs(x_safe)
        # Equivalente exato em float32 a log1p(|x|) usando apenas operadores ONNX nativos (Log, Where, Square)
        log1p_abs = tf.where(
            abs_x < 1e-4,
            abs_x - 0.5 * tf.square(abs_x),
            tf.math.log(1.0 + abs_x),
        )
        x_cov = tf.math.sign(x_safe) * log1p_abs
        x_pre = tf.where(tf.constant(cov_mask_np, dtype=tf.bool), x_cov, x_safe)

        z = (x_pre - tf.constant(mean_np, dtype=tf.float32)) / tf.constant(std_np, dtype=tf.float32)
        x_cont_norm = tf.where(is_valid, z, tf.zeros_like(z))

        cat_encoded_tensors = []
        for i, (k_vals, k_idxs, unk_idx) in enumerate(cat_tables):
            col_val = raw_categorical[:, i : i + 1]
            matches = tf.equal(col_val, tf.constant([k_vals], dtype=tf.int32))
            any_match = tf.reduce_any(matches, axis=1, keepdims=True)
            pos = tf.argmax(tf.cast(matches, tf.int32), axis=1)
            mapped = tf.expand_dims(tf.gather(tf.constant(k_idxs, dtype=tf.int32), pos), axis=1)
            encoded = tf.where(any_match, mapped, tf.constant(unk_idx, dtype=tf.int32))
            cat_encoded_tensors.append(encoded)

        kappa_scaled = raw_kappa_lambda / tf.constant(kappa_norm, dtype=tf.float32)
        probs = model([x_cont_norm] + cat_encoded_tensors + [kappa_scaled], training=False)
        return {"class_probs": probs}

    patch_tf2onnx()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(
        forward_end2end,
        input_signature=specs,
        opset=int(opset),
        output_path=str(onnx_path),
    )

    if metadata:
        _attach_onnx_metadata(onnx_path, metadata)

    return {
        "file": onnx_path.name,
        "opset": int(opset),
        "inputs": [
            {"name": "raw_continuous", "shape": ["batch", len(cont_cols)], "dtype": "float32", "columns": cont_cols},
            {"name": "raw_categorical", "shape": ["batch", len(cat_cols)], "dtype": "int32", "columns": cat_cols},
            {
                "name": "raw_kappa_lambda",
                "shape": ["batch", 1],
                "dtype": "float32",
                "description": f"Physical kappa_lambda (internally divided by kappa_norm={kappa_norm})",
            },
        ],
        "outputs": [{"name": "class_probs", "shape": ["batch", int(model.output_shape[-1])], "dtype": "float32"}],
    }


def verify_onnx_parity(
    model: keras.Model,
    preprocessed_onnx_path: Path,
    end2end_onnx_path: Path,
    scaler: FeatureScaler,
    cat_preprocessor: CategoricalPreprocessor,
    cfg: dict[str, Any],
    n_test_samples: int = 256,
    seed: int = 42,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Executa checagem automática de paridade numérica entre Keras e ambos os grafos ONNX via `onnxruntime`."""
    import onnxruntime as ort

    from src.data.preprocessing import apply_physical_preprocessing

    rng = np.random.default_rng(seed)
    cont_cols = list(scaler.features)
    cat_cols = list(cat_preprocessor.categorical_features)
    knorm = float(cfg["physics"]["kappa_norm"])
    kgrid = cfg["physics"].get("kappa_grid", [0.0, 1.0, 2.45, 5.0])

    # Amostra sintética realista cobrindo objetos ausentes, MET cov de alto alcance e todos os κ_λ
    raw_cont = (rng.normal(0.0, 1.0, size=(n_test_samples, len(cont_cols))).astype(np.float32) * scaler.std + scaler.mean)
    raw_cont[:, :4] = np.abs(raw_cont[:, :4]) + 15.0  # energias positivas

    raw_cat_cols = {}
    raw_cat_mat = np.zeros((n_test_samples, len(cat_cols)), dtype=np.int32)
    for j, c in enumerate(cat_cols):
        known = list(cat_preprocessor.encoders[c].value_to_index.keys())
        sampled = rng.choice(known, size=n_test_samples).astype(np.int32)
        raw_cat_cols[c] = sampled
        raw_cat_mat[:, j] = sampled

    # Injeta alguns valores sentinelas (-9999) para testar a limpeza interna do grafo End-to-End
    raw_cont[0, 10] = -9999.0
    raw_kappa = rng.choice(np.asarray(kgrid, dtype=np.float32), size=(n_test_samples, 1))

    # Caminho 1: Pipeline Python + Keras
    df_raw = pd.DataFrame(raw_cont, columns=cont_cols)
    for c, arr in raw_cat_cols.items():
        df_raw[c] = arr
    for sv in cfg["features"].get("sentinels", [-99999.0, -9999.0, -999.0]):
        df_raw[cont_cols] = df_raw[cont_cols].replace(sv, np.nan)

    df_prep = apply_physical_preprocessing(
        df_raw.copy(),
        continuous_features=cont_cols,
        cov_cols=cfg["features"].get("cov_cols", []),
        missing_masks=cfg["features"].get("missing_object_masks", {}),
        mask_bjet1_always_on_no_pair=bool(cfg["features"].get("mask_bjet1_always_on_no_pair", False)),
    )
    x_cont_py = scaler.transform(df_prep)
    x_cat_py = cat_preprocessor.transform(df_prep)
    keras_inputs = make_inputs(x_cont_py, x_cat_py, raw_kappa.ravel(), kappa_norm=knorm)

    # Avalia a referência Keras em CPU IEEE-754 FP32 (evitando truncamento TF32 de 10 bits das GPUs Ampere/A100!)
    ordered_tensors = [tf.constant(keras_inputs["x_continuous"], dtype=tf.float32)]
    for c in cat_cols:
        ordered_tensors.append(tf.constant(keras_inputs[f"cat_{short_cat_name(c)}"], dtype=tf.int32))
    ordered_tensors.append(tf.constant(keras_inputs["kappa_lambda"], dtype=tf.float32))
    with tf.device("/CPU:0"):
        probs_keras = keras.ops.convert_to_numpy(model(ordered_tensors, training=False)).astype(np.float32)

    probs_default_dev = model.predict(keras_inputs, batch_size=256, verbose=0).astype(np.float32)

    # Caminho 2: ONNX pré-processado (CPU FP32)
    sess_pre = ort.InferenceSession(str(preprocessed_onnx_path), providers=["CPUExecutionProvider"])
    ort_pre_inputs = {"x_continuous": keras_inputs["x_continuous"], "kappa_lambda": keras_inputs["kappa_lambda"]}
    for c in cat_cols:
        sc = short_cat_name(c)
        ort_pre_inputs[f"cat_{sc}"] = keras_inputs[f"cat_{sc}"]
    probs_onnx_pre = sess_pre.run(["class_probs"], ort_pre_inputs)[0]

    # Caminho 3: ONNX End-to-End (direto nos arrays brutos com sentinela e objetos ausentes!)
    sess_e2e = ort.InferenceSession(str(end2end_onnx_path), providers=["CPUExecutionProvider"])
    probs_onnx_e2e = sess_e2e.run(
        ["class_probs"],
        {
            "raw_continuous": raw_cont,
            "raw_categorical": raw_cat_mat,
            "raw_kappa_lambda": raw_kappa,
        },
    )[0]

    diff_pre = np.abs(probs_keras - probs_onnx_pre)
    diff_e2e = np.abs(probs_keras - probs_onnx_e2e)
    diff_onnx_pair = np.abs(probs_onnx_pre - probs_onnx_e2e)
    diff_gpu_tf32 = np.abs(probs_default_dev - probs_onnx_e2e)

    max_diff_pre = float(diff_pre.max())
    max_diff_e2e = float(diff_e2e.max())
    max_diff_pair = float(diff_onnx_pair.max())

    if max_diff_pre > atol or max_diff_e2e > atol:
        raise AssertionError(
            f"Divergência numérica FP32 acima da tolerância ({atol}) na exportação ONNX! "
            f"max_diff_preprocessed={max_diff_pre:.3e}, max_diff_end2end={max_diff_e2e:.3e}"
        )

    return {
        "n_test_samples": int(n_test_samples),
        "tolerance_fp32": float(atol),
        "preprocessed_onnx_max_abs_diff": max_diff_pre,
        "preprocessed_onnx_mean_abs_diff": float(diff_pre.mean()),
        "end2end_raw_onnx_max_abs_diff": max_diff_e2e,
        "end2end_raw_onnx_mean_abs_diff": float(diff_e2e.mean()),
        "onnx_pre_vs_end2end_max_abs_diff": max_diff_pair,
        "default_device_vs_onnx_fp32_max_abs_diff": float(diff_gpu_tf32.max()),
        "status": "PASSED",
    }


class ONNXModelPredictor:
    """Preditor leve baseado exclusivamente em `onnxruntime` (não requer TensorFlow/Keras instalado!)."""

    def __init__(self, bundle_or_onnx_path: str | Path, use_end2end_raw: bool = True):
        import onnx
        import onnxruntime as ort

        p = Path(bundle_or_onnx_path)
        if p.is_dir():
            manifest_path = p / "bundle_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                fname = (
                    manifest["artifacts"]["onnx_end2end_raw"]
                    if use_end2end_raw
                    else manifest["artifacts"]["onnx_preprocessed"]
                )
                onnx_file = p / fname
            else:
                suffix = "*_end2end_raw.onnx" if use_end2end_raw else "*.onnx"
                matches = sorted(p.glob(suffix))
                if not matches:
                    raise FileNotFoundError(f"Nenhum arquivo ONNX encontrado em {p}")
                onnx_file = matches[0]
        else:
            onnx_file = p

        self.onnx_path = onnx_file
        self.session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        proto = onnx.load(str(onnx_file))
        self.metadata = {prop.key: prop.value for prop in proto.metadata_props}
        self.continuous_features: list[str] = json.loads(self.metadata.get("continuous_features", "[]"))
        self.categorical_features: list[str] = json.loads(self.metadata.get("categorical_features", "[]"))
        self.class_labels: dict[str, str] = json.loads(self.metadata.get("class_labels", "{}"))

    def predict_raw_dataframe(self, df: pd.DataFrame, kappa_lambda: float | np.ndarray) -> np.ndarray:
        """Prediz `[P(DY), P(ttbar), P(HH)]` diretamente sobre um DataFrame bruto do Parquet usando o ONNX End-to-End."""
        n = len(df)
        raw_cont = df[self.continuous_features].to_numpy(np.float32)
        raw_cat = np.column_stack(
            [np.round(df[c].to_numpy(np.float64)).astype(np.int32) for c in self.categorical_features]
        )
        if np.ndim(kappa_lambda) == 0:
            raw_k = np.full((n, 1), float(kappa_lambda), dtype=np.float32)
        else:
            raw_k = np.asarray(kappa_lambda, dtype=np.float32).reshape(n, 1)

        return self.session.run(
            ["class_probs"],
            {
                "raw_continuous": raw_cont,
                "raw_categorical": raw_cat,
                "raw_kappa_lambda": raw_k,
            },
        )[0]


def export_complete_model_bundle(
    model_name: str,
    model: keras.Model,
    scaler: FeatureScaler,
    cat_preprocessor: CategoricalPreprocessor,
    cfg: dict[str, Any],
    export_root_dir: str | Path,
    history_dict: dict[str, list[float]] | None = None,
    meta_dict: dict[str, Any] | None = None,
    opset: int = 17,
    verify_parity: bool = True,
) -> dict[str, Any]:
    """Exporta o pacote completo e autossuficiente de um modelo treinado (`Keras + H5 + SavedModel + 2x ONNX + NPZ + JSON`)."""
    import yaml

    bundle_dir = Path(export_root_dir) / model_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    cont_cols = list(scaler.features)
    cat_cols = list(cat_preprocessor.categorical_features)
    knorm = float(cfg["physics"]["kappa_norm"])
    kgrid = [float(k) for k in cfg["physics"].get("kappa_grid", [0.0, 1.0, 2.45, 5.0])]
    class_labels = {str(k): str(v) for k, v in cfg["physics"]["class_labels"].items()}

    # 1. Keras v3 (.keras) + Pesos HDF5 (.weights.h5)
    keras_path = bundle_dir / f"model_{model_name}.keras"
    weights_path = bundle_dir / f"{model_name}.weights.h5"
    model.save(keras_path)
    model.save_weights(weights_path)

    # 2. TensorFlow SavedModel nativo (para C++ / TF Serving)
    saved_model_dir = bundle_dir / "saved_model"
    model.export(str(saved_model_dir), format="tf_saved_model", verbose=False)

    # 3. Arquitetura detalhada + todos os parâmetros aprendidos em NumPy (.npz)
    npz_path = bundle_dir / f"learned_parameters_{model_name}.npz"
    params_info = export_learned_parameters_npz(model, npz_path)

    arch_path = bundle_dir / f"architecture_{model_name}.json"
    with open(arch_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "keras_class": model.__class__.__name__,
                "total_params": int(model.count_params()),
                "keras_config": json.loads(model.to_json()),
                "layer_summary": params_info["layers"],
            },
            f,
            indent=2,
        )

    # 4. Especificação completa de pré-processamento + CSV/JSON + YAML congelado
    scaler_csv = bundle_dir / "normalization_train_only.csv"
    enc_json = bundle_dir / "config_encoders.json"
    scaler.save_csv(scaler_csv)
    cat_preprocessor.save_json(
        enc_json,
        continuous_features=cont_cols,
        kappa_norm=knorm,
        kappa_grid=kgrid,
    )

    prep_spec_path = bundle_dir / "preprocessing_spec.json"
    prep_spec = {
        "continuous_features": cont_cols,
        "categorical_features": cat_cols,
        "feature_mean": {f: float(m) for f, m in zip(cont_cols, scaler.mean)},
        "feature_std": {f: float(s) for f, s in zip(cont_cols, scaler.std)},
        "categorical_encoders": {c: cat_preprocessor.encoders[c].to_dict() for c in cat_cols},
        "cat_vocab_sizes": cat_preprocessor.vocab_sizes,
        "sentinels": cfg["features"].get("sentinels", [-99999.0, -9999.0, -999.0]),
        "cov_cols_signed_log1p": cfg["features"].get("cov_cols", []),
        "missing_object_masks": cfg["features"].get("missing_object_masks", {}),
        "lbn_particles": cfg["features"].get("lbn_particles", []),
        "kappa_norm": knorm,
        "kappa_grid": kgrid,
        "class_labels": class_labels,
    }
    with open(prep_spec_path, "w", encoding="utf-8") as f:
        json.dump(prep_spec, f, indent=2)

    cfg_yaml_path = bundle_dir / "training_config_full.yaml"
    with open(cfg_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    if history_dict is not None:
        df_h = pd.DataFrame(history_dict)
        df_h.index.name = "epoch_0based"
        df_h.to_csv(bundle_dir / f"history_{model_name}.csv")
        with open(bundle_dir / f"history_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=2)

    if meta_dict is not None:
        with open(bundle_dir / f"meta_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

    # 5. Exportação dupla em ONNX (pré-processado + End-to-End Físico Bruto)
    onnx_meta_props = {
        "project": "CMS Run-3 HH->bbtautau (ggF) AffinePNN vs AN-25-103",
        "model_name": model_name,
        "continuous_features": json.dumps(cont_cols),
        "categorical_features": json.dumps(cat_cols),
        "class_labels": json.dumps(class_labels),
        "kappa_norm": str(knorm),
        "kappa_grid": json.dumps(kgrid),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    onnx_pre_path = bundle_dir / f"{model_name}.onnx"
    onnx_e2e_path = bundle_dir / f"{model_name}_end2end_raw.onnx"

    onnx_pre_info = export_preprocessed_onnx(
        model=model,
        onnx_path=onnx_pre_path,
        n_continuous=len(cont_cols),
        cat_vocab_sizes=cat_preprocessor.vocab_sizes,
        opset=opset,
        metadata={**onnx_meta_props, "onnx_variant": "preprocessed_inputs"},
    )
    onnx_e2e_info = export_end2end_raw_onnx(
        model=model,
        onnx_path=onnx_e2e_path,
        scaler=scaler,
        cat_preprocessor=cat_preprocessor,
        cfg=cfg,
        opset=opset,
        metadata={**onnx_meta_props, "onnx_variant": "end2end_raw_physics_inputs"},
    )

    parity_report = None
    if verify_parity:
        parity_report = verify_onnx_parity(
            model=model,
            preprocessed_onnx_path=onnx_pre_path,
            end2end_onnx_path=onnx_e2e_path,
            scaler=scaler,
            cat_preprocessor=cat_preprocessor,
            cfg=cfg,
            seed=int(cfg.get("seed", 42)),
        )

    # 6. Manifesto auditável com SHA-256 de todos os arquivos gerados
    tracked_files = [
        keras_path,
        weights_path,
        onnx_pre_path,
        onnx_e2e_path,
        npz_path,
        arch_path,
        prep_spec_path,
        scaler_csv,
        enc_json,
        cfg_yaml_path,
    ]
    checksums = {p.name: {"sha256": _sha256_file(p), "size_bytes": p.stat().st_size} for p in tracked_files if p.exists()}

    manifest = {
        "model_name": model_name,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir),
        "total_params": int(model.count_params()),
        "training_meta": meta_dict,
        "environment": _collect_environment_versions(),
        "artifacts": {
            "keras_model": keras_path.name,
            "hdf5_weights": weights_path.name,
            "tf_saved_model_dir": saved_model_dir.name,
            "onnx_preprocessed": onnx_pre_path.name,
            "onnx_end2end_raw": onnx_e2e_path.name,
            "learned_parameters_npz": npz_path.name,
            "architecture_json": arch_path.name,
            "preprocessing_spec_json": prep_spec_path.name,
            "scaler_csv": scaler_csv.name,
            "encoders_json": enc_json.name,
            "config_yaml": cfg_yaml_path.name,
        },
        "onnx_signatures": {
            "preprocessed": onnx_pre_info,
            "end2end_raw": onnx_e2e_info,
        },
        "onnx_parity_verification": parity_report,
        "checksums": checksums,
    }

    manifest_path = bundle_dir / "bundle_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
