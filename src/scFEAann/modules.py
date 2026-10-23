"""Human scFEA module annotation and stoichiometry validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import PACKAGE_DATA_DIR


DEFAULT_MODULE_GENE_FILE = "module_gene_m168.csv"
DEFAULT_MODULE_INFO_FILE = "Human_M168_information.symbols.csv"
DEFAULT_STOICHIOMETRY_MATRIX = "cmMat_c70_m168.csv"
DEFAULT_COMPOUND_NAME_FILE = "cName_c70_m168.csv"


def _resolve(data_dir: str | Path | None, filename: str) -> Path:
    base = PACKAGE_DATA_DIR if data_dir is None else Path(data_dir)
    path = Path(filename)
    return path if path.is_absolute() else base / path


def _split_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [
        token.strip()
        for token in str(value).split("+")
        if token.strip() and token.strip().lower() != "nan"
    ]


def _split_names(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [
        token.strip()
        for token in str(value).split("+")
        if token.strip() and token.strip().lower() != "nan"
    ]


def get_module_annotation(
    *,
    data_dir: str | Path | None = None,
    module_gene_file: str = DEFAULT_MODULE_GENE_FILE,
    module_info_file: str = DEFAULT_MODULE_INFO_FILE,
    stoichiometry_matrix: str = DEFAULT_STOICHIOMETRY_MATRIX,
    compound_name_file: str = DEFAULT_COMPOUND_NAME_FILE,
    strict: bool = True,
) -> pd.DataFrame:
    """Return the human module-to-reaction mapping and validation flags.

    The annotation combines module genes, reaction input/output names and KEGG
    identifiers, the supermodule identifier, and the corresponding signed
    stoichiometry row. A negative stoichiometric coefficient denotes an input
    and a positive coefficient denotes an output.
    """
    module_path = _resolve(data_dir, module_gene_file)
    info_path = _resolve(data_dir, module_info_file)
    stoichiometry_path = _resolve(data_dir, stoichiometry_matrix)
    compound_path = _resolve(data_dir, compound_name_file)
    for path in (module_path, info_path, stoichiometry_path, compound_path):
        if not path.exists():
            raise ValueError(f"module annotation file does not exist: {path}")

    module_genes = pd.read_csv(module_path, index_col=0)
    info = pd.read_csv(info_path, index_col=0)
    stoichiometry = pd.read_csv(stoichiometry_path, header=None).to_numpy(
        dtype=np.int64
    )
    compounds = pd.read_csv(compound_path, header=None, dtype=str)
    compound_names = compounds.iloc[0].astype(str).tolist()
    compound_ids = compounds.iloc[1].astype(str).tolist()
    if len(set(compound_ids)) != len(compound_ids):
        raise ValueError("compound identifiers must be unique")
    id_to_row = {
        compound_id: index
        for index, compound_id in enumerate(compound_ids)
    }
    if stoichiometry.shape != (len(compound_ids), module_genes.shape[0]):
        raise ValueError(
            "stoichiometry matrix shape does not match compounds and modules"
        )
    if not module_genes.index.equals(info.index):
        raise ValueError(
            "module gene and module information files must have identical row order"
        )
    if not module_genes.index.is_unique:
        raise ValueError("module identifiers must be unique")

    records: list[dict[str, object]] = []
    mismatches: list[str] = []
    for module_index, (module_id, module_row) in enumerate(
        module_genes.iterrows()
    ):
        info_row = info.loc[module_id]
        input_names = _split_names(info_row["Compound_IN_name"])
        input_ids = _split_ids(info_row["Compound_IN_ID"])
        output_names = _split_names(info_row["Compound_OUT_name"])
        output_ids = _split_ids(info_row["Compound_OUT_ID"])
        if len(input_names) != len(input_ids):
            raise ValueError(f"input name/ID count mismatch for {module_id}")
        if len(output_names) != len(output_ids):
            raise ValueError(f"output name/ID count mismatch for {module_id}")

        expected = np.zeros(len(compound_ids), dtype=np.int64)
        unmodeled: list[str] = []
        for compound_id in input_ids:
            if compound_id in id_to_row:
                expected[id_to_row[compound_id]] -= 1
            else:
                unmodeled.append(f"input:{compound_id}")
        for compound_id in output_ids:
            if compound_id in id_to_row:
                expected[id_to_row[compound_id]] += 1
            else:
                unmodeled.append(f"output:{compound_id}")
        actual = stoichiometry[:, module_index]
        matches = bool(np.array_equal(expected, actual))
        if not matches:
            mismatches.append(str(module_id))
        negative = [
            compound_names[index]
            for index in np.flatnonzero(actual < 0)
        ]
        positive = [
            compound_names[index]
            for index in np.flatnonzero(actual > 0)
        ]
        genes = [
            str(gene)
            for gene in module_row.dropna().tolist()
            if str(gene).strip()
        ]
        records.append(
            {
                "module_id": str(module_id),
                "module_number": int(info_row["Module_id"]),
                "reaction": (
                    f"{' + '.join(input_names)} -> "
                    f"{' + '.join(output_names)}"
                ),
                "input_names": " + ".join(input_names),
                "input_kegg_ids": "+".join(input_ids),
                "output_names": " + ".join(output_names),
                "output_kegg_ids": "+".join(output_ids),
                "supermodule_id": int(info_row["Supermodule_id"]),
                "module_genes": ";".join(genes),
                "n_genes": len(genes),
                "stoichiometry_inputs": ";".join(negative),
                "stoichiometry_outputs": ";".join(positive),
                "unmodeled_reference_ids": ";".join(unmodeled),
                "external_input": "C00000" in input_ids,
                "external_output": "C00000" in output_ids,
                "modeled_in_balance": bool(np.any(actual)),
                "stoichiometry_match": matches,
            }
        )
    if strict and mismatches:
        raise ValueError(
            "module information and stoichiometry matrix disagree for: "
            + ", ".join(mismatches)
        )
    annotation = pd.DataFrame.from_records(records, index="module_id")
    annotation.index.name = "module_id"
    return annotation
