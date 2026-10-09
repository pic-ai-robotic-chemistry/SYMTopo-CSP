import fractions
import re


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        float(fractions.Fraction(s))
        return True
    except ValueError:
        return False


def normalize_site_label(raw_label, element, atom_id):
    if raw_label is None:
        raw_label = ""
    label = str(raw_label).strip()

    if label == "" or label.lower() == "unknown":
        label = f"{element}{atom_id+1}"

    # Keep CIF labels limited to alphanumeric characters and underscores.
    label = re.sub(r"[^A-Za-z0-9_]", "_", label)
    if label == "":
        label = f"{element}{atom_id+1}"
    return label


def make_symm_copy_label(base_label, symm_op_id, ensure_unique=True):
    if not ensure_unique or symm_op_id == 0:
        return base_label
    return f"{base_label}_s{symm_op_id+1}"
