"""Ligand identity helpers."""

from __future__ import annotations

from mdprep.structure.models import AtomRecord, ResidueRecord
from mdprep.structure.pdb import infer_element


def ligand_identity_dict(
    *,
    ligand_id: str,
    selector: dict[str, object],
    residue: ResidueRecord,
    net_charge: int,
    multiplicity: int,
    charge_method: str,
    atom_types: str,
) -> dict[str, object]:
    return {
        "ligand_id": ligand_id,
        "selector": selector,
        "atom_count": len(residue.atoms),
        "atom_names": [atom.name for atom in residue.atoms],
        "elements": [atom_element(atom) for atom in residue.atoms],
        "residue_name": residue.id.resname,
        "residue_number": residue.id.resid,
        "chain_id": residue.id.chain_id,
        "icode": residue.id.icode,
        "net_charge": net_charge,
        "multiplicity": multiplicity,
        "charge_method": charge_method,
        "atom_types": atom_types,
    }


def atom_element(atom: AtomRecord) -> str:
    return atom.element or infer_element(atom.name) or ""

