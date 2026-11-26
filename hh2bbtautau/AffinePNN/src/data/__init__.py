"""Módulos de leitura de Parquets, pré-processamento físico, pesos e quotas."""

from src.data.parquet_dataset import (
    check_feature_leakage_names,
    check_split_duplicates,
    clean_and_check_data_quality,
    load_all_parquets,
    read_parquet_limited,
    row_hashes,
    short_cat_name,
)
from src.data.preprocessing import (
    CategoricalEncoder,
    CategoricalPreprocessor,
    FeatureScaler,
    PreparedData,
    apply_physical_preprocessing,
    check_four_vectors,
    get_lbn_column_indices,
    make_inputs,
    prepare_datasets_from_config,
    stratified_split,
)
from src.data.sampler import (
    QuotaBatchGenerator,
    make_random_label_generator,
    pool_report,
    verify_generator_quotas,
)
from src.data.weights import (
    aggregate_weight,
    class_weight_summary,
    ess,
    make_eval_weights,
    oversampling_frequency_weights,
    subprocess_weight_summary,
    weight_report,
)

__all__ = [
    "CategoricalEncoder",
    "CategoricalPreprocessor",
    "FeatureScaler",
    "PreparedData",
    "QuotaBatchGenerator",
    "aggregate_weight",
    "apply_physical_preprocessing",
    "check_feature_leakage_names",
    "check_four_vectors",
    "check_split_duplicates",
    "class_weight_summary",
    "clean_and_check_data_quality",
    "ess",
    "get_lbn_column_indices",
    "load_all_parquets",
    "make_eval_weights",
    "make_inputs",
    "make_random_label_generator",
    "oversampling_frequency_weights",
    "pool_report",
    "prepare_datasets_from_config",
    "read_parquet_limited",
    "row_hashes",
    "short_cat_name",
    "stratified_split",
    "subprocess_weight_summary",
    "verify_generator_quotas",
    "weight_report",
]
