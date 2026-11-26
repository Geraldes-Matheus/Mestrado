"""Camadas customizadas serializáveis, construtores de modelos e exportação completa (Keras/SavedModel/ONNX)."""

from src.models.affine_dnn import (
    build_model,
    build_model_from_config,
    extract_affine_modulation,
    load_saved_model,
)
from src.models.affine_layers import (
    CUSTOM_OBJECTS,
    AffineConditioning,
    AsinhScale,
    FourVectorsFromFeatures,
    LorentzBoostNetwork,
    TakeColumns,
)
from src.models.embeddings import build_tabular_and_categorical_inputs
from src.models.export import (
    ONNXModelPredictor,
    export_complete_model_bundle,
    export_end2end_raw_onnx,
    export_learned_parameters_npz,
    export_preprocessed_onnx,
    verify_onnx_parity,
)

__all__ = [
    "CUSTOM_OBJECTS",
    "AffineConditioning",
    "AsinhScale",
    "FourVectorsFromFeatures",
    "LorentzBoostNetwork",
    "ONNXModelPredictor",
    "TakeColumns",
    "build_model",
    "build_model_from_config",
    "build_tabular_and_categorical_inputs",
    "export_complete_model_bundle",
    "export_end2end_raw_onnx",
    "export_learned_parameters_npz",
    "export_preprocessed_onnx",
    "extract_affine_modulation",
    "load_saved_model",
    "verify_onnx_parity",
]
