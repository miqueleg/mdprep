"""Pure notebook helpers for the mdprep Google Colab workflow.

There is deliberately no widget application or external web interface here.
Google Colab executes these functions from ordinary cells and renders molecular
structures directly in the cell output.
"""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import py3Dmol
import requests
import yaml
from IPython.display import HTML, display

WORKDIR = Path(os.environ.get("MDPREP_COLAB_WORKDIR", "/content/mdprep_colab"))
WORKDIR.mkdir(parents=True, exist_ok=True)
WATERS = {"HOH", "WAT", "TIP3", "TP3", "SOL"}
METAL_ELEMENTS = {
    "LI",
    "NA",
    "K",
    "RB",
    "CS",
    "MG",
    "CA",
    "SR",
    "BA",
    "ZN",
    "FE",
    "MN",
    "CU",
    "CO",
    "NI",
    "CD",
    "HG",
    "CR",
    "V",
    "MO",
    "W",
}


@dataclass(frozen=True, order=True)
class ComponentKey:
    chain: str
    resname: str
    resid: int
    icode: str = ""

    @property
    def label(self) -> str:
        return f"{self.chain or '_'}:{self.resname}:{self.resid}{self.icode}"


def atom_record(line: str) -> bool:
    return line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54


def component_key(line: str) -> ComponentKey:
    return ComponentKey(
        chain=line[21:22].strip(),
        resname=line[17:20].strip().upper(),
        resid=int(line[22:26]),
        icode=line[26:27].strip(),
    )


def element(line: str) -> str:
    explicit = line[76:78].strip().upper() if len(line) >= 78 else ""
    if explicit:
        return explicit
    return re.sub(r"[^A-Za-z]", "", line[12:16]).strip()[:2].upper()


def xyz(line: str) -> np.ndarray:
    return np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def group_components(text: str) -> dict[ComponentKey, list[str]]:
    groups: dict[ComponentKey, list[str]] = defaultdict(list)
    for line in text.splitlines():
        if line.startswith("HETATM"):
            groups[component_key(line)].append(line)
    return dict(groups)


def is_metal(lines: list[str]) -> bool:
    return len(lines) == 1 and element(lines[0]) in METAL_ELEMENTS


def formula_from_lines(lines: list[str]) -> str:
    counts = Counter(element(line).capitalize() for line in lines if atom_record(line))
    order = ["C", "H"] + sorted(item for item in counts if item not in {"C", "H"})
    return "".join(
        f"{item}{counts[item] if counts[item] != 1 else ''}"
        for item in order
        if counts[item]
    )


def renumber_pdb(text: str) -> str:
    output: list[str] = []
    serial = 1
    for line in text.splitlines():
        if line.startswith("CONECT"):
            continue
        if atom_record(line):
            output.append(f"{line[:6]}{serial:5d}{line[11:]}")
            serial += 1
    output.append("END")
    return "\n".join(output) + "\n"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def show_structure(text: str, title: str, *, height: int = 600) -> None:
    """Display a structure directly in the current Colab cell output."""
    print(title)
    view = py3Dmol.view(width=950, height=height)
    view.addModel(text, "pdb")
    view.setStyle({"hetflag": False}, {"cartoon": {"color": "spectrum"}})
    view.addStyle(
        {"hetflag": True},
        {"stick": {"colorscheme": "greenCarbon", "radius": 0.2}},
    )
    view.addStyle({"resn": list(WATERS)}, {"sphere": {"radius": 0.22, "color": "red"}})
    view.addStyle(
        {"elem": list(METAL_ELEMENTS)},
        {"sphere": {"radius": 0.7, "color": "orange"}},
    )
    view.zoomTo()
    display(HTML(view._make_html()))


def download_rcsb_pdb(pdb_id: str) -> str:
    code = pdb_id.strip().upper()
    if not re.fullmatch(r"[0-9A-Za-z]{4}", code):
        raise ValueError("PDB_ID must contain exactly four letters/numbers")
    response = requests.get(f"https://files.rcsb.org/download/{code}.pdb", timeout=90)
    response.raise_for_status()
    if not any(atom_record(line) for line in response.text.splitlines()):
        raise ValueError(f"RCSB returned no readable PDB atoms for {code}")
    return response.text


def print_inventory(text: str) -> None:
    chains = sorted(
        {
            line[21:22].strip() or "_"
            for line in text.splitlines()
            if line.startswith("ATOM  ")
        }
    )
    print("Protein chains:", ", ".join(chains) or "none")
    print("\nNon-protein components:")
    groups = group_components(text)
    for key in sorted(groups):
        if key.resname in WATERS:
            continue
        suffix = " [metal]" if is_metal(groups[key]) else ""
        print(
            f"  {key.label:18s} {len(groups[key]):3d} atom records  "
            f"observed={formula_from_lines(groups[key])}{suffix}"
        )
    waters = sum(1 for key in groups if key.resname in WATERS)
    print(f"\nCrystal-water residues: {waters}")


def parse_component_labels(value: str) -> set[ComponentKey]:
    selected: set[ComponentKey] = set()
    for raw_label in value.split(","):
        label = raw_label.strip()
        if not label:
            continue
        fields = label.split(":")
        if len(fields) != 3:
            raise ValueError(f"Invalid component label {label!r}; use chain:RES:resid")
        chain, resname, resid = fields
        selected.add(
            ComponentKey("" if chain == "_" else chain, resname.upper(), int(resid))
        )
    return selected


def _resolve_altlocs(lines: list[str]) -> tuple[list[str], int]:
    buckets: dict[tuple[str, str, str, int, str, str], list[tuple[int, str]]] = (
        defaultdict(list)
    )
    for index, line in enumerate(lines):
        if not atom_record(line):
            continue
        key = component_key(line)
        identity = (
            line[:6],
            key.chain,
            key.resname,
            key.resid,
            key.icode,
            line[12:16].strip(),
        )
        buckets[identity].append((index, line))
    keep: set[int] = set()
    removed = 0
    for candidates in buckets.values():
        winner = min(
            candidates,
            key=lambda item: (
                -float(item[1][54:60].strip() or 0.0),
                item[1][16:17].strip() != "",
                item[1][16:17],
            ),
        )
        keep.add(winner[0])
        removed += len(candidates) - 1
    output = [
        f"{line[:16]} {line[17:]}" if atom_record(line) else line
        for index, line in enumerate(lines)
        if not atom_record(line) or index in keep
    ]
    return output, removed


def clean_structure(
    text: str,
    *,
    protein_chains: str,
    component_labels: str,
    water_selection: str,
    water_cutoff_angstrom: float,
    confirmed: bool,
) -> str:
    """Apply only the cleanup choices explicitly confirmed in the Colab form."""
    if not confirmed:
        raise ValueError("Set CONFIRM_CLEANUP=True after reviewing the inventory")
    chains = {item.strip() for item in protein_chains.split(",") if item.strip()}
    if not chains:
        raise ValueError("Select at least one protein chain")
    selected = parse_component_labels(component_labels)
    groups = group_components(text)
    missing = selected - set(groups)
    if missing:
        raise ValueError(
            f"Selected components were not found: {sorted(item.label for item in missing)}"
        )
    anchors = [
        xyz(line) for key in selected for line in groups[key] if element(line) != "H"
    ]
    kept: list[str] = []
    for line in text.splitlines():
        if line.startswith("ATOM  ") and line[21:22].strip() in chains:
            kept.append(line)
            continue
        if not line.startswith("HETATM"):
            continue
        key = component_key(line)
        if key in selected:
            kept.append(line)
        elif key.resname in WATERS and key.chain in chains:
            if water_selection == "all selected-chain waters":
                kept.append(line)
            elif water_selection == "within cutoff" and anchors:
                distance = min(np.linalg.norm(xyz(line) - point) for point in anchors)
                if distance <= water_cutoff_angstrom:
                    kept.append(line)
    kept, removed_altlocs = _resolve_altlocs(kept)
    cleaned = renumber_pdb("\n".join(kept))
    selected_groups = {
        key: lines
        for key, lines in group_components(cleaned).items()
        if key.resname not in WATERS
    }
    water_count = sum(1 for key in group_components(cleaned) if key.resname in WATERS)
    print(
        f"Kept chains {sorted(chains)}, {len(selected_groups)} non-water components, "
        f"and {water_count} waters. Resolved {removed_altlocs} alternate locations."
    )
    return cleaned


# ---- Component chemistry -------------------------------------------------


def _choice(prompt: str, choices: tuple[str, ...]) -> str:
    allowed = {choice.lower(): choice for choice in choices}
    while True:
        value = input(f"{prompt} [{'/'.join(choices)}]: ").strip().lower()
        if value in allowed:
            return allowed[value]
        print("Please enter exactly one of:", ", ".join(choices))


def _required_int(prompt: str, *, minimum: int | None = None) -> int:
    while True:
        try:
            value = int(input(f"{prompt}: ").strip())
        except ValueError:
            print("An explicit integer is required.")
            continue
        if minimum is not None and value < minimum:
            print(f"The value must be at least {minimum}.")
            continue
        return value


def _required_float(prompt: str, *, minimum: float | None = None) -> float:
    while True:
        try:
            value = float(input(f"{prompt}: ").strip())
        except ValueError:
            print("An explicit number is required.")
            continue
        if minimum is not None and value < minimum:
            print(f"The value must be at least {minimum}.")
            continue
        return value


def _yes(prompt: str) -> bool:
    return _choice(prompt, ("yes", "no")) == "yes"


def _residue_selector_from_label(label: str) -> dict[str, Any]:
    """Parse ``chain:RES:resid[icode]`` without guessing residue identity."""
    fields = [item.strip() for item in label.split(":")]
    if len(fields) != 3:
        raise ValueError(
            f"Invalid residue selector {label!r}; use chain:RES:resid, for example "
            "A:HIS:134"
        )
    chain, resname, residue_token = fields
    match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", residue_token)
    if not match:
        raise ValueError(f"Invalid residue number/insertion code in {label!r}")
    return {
        "chain": "" if chain == "_" else chain,
        "resname": resname.upper(),
        "resid": int(match.group(1)),
        "icode": match.group(2) or None,
    }


def _atom_selector_from_label(label: str) -> dict[str, Any]:
    """Parse ``chain:RES:resid[icode]:ATOM`` for an explicit MCPB bond."""
    fields = [item.strip() for item in label.split(":")]
    if len(fields) != 4:
        raise ValueError(
            f"Invalid atom selector {label!r}; use chain:RES:resid:ATOM, for "
            "example A:HIS:134:NE2"
        )
    residue = _residue_selector_from_label(":".join(fields[:3]))
    if not fields[3]:
        raise ValueError(f"Atom name is missing from {label!r}")
    return {**residue, "atom_name": fields[3]}


def _selector_label(selector: dict[str, Any], *, include_atom: bool = True) -> str:
    residue = (
        f"{selector['chain'] or '_'}:{selector['resname']}:"
        f"{selector['resid']}{selector.get('icode') or ''}"
    )
    return f"{residue}:{selector['atom_name']}" if include_atom else residue


def _matching_atom_lines(text: str, selector: dict[str, Any]) -> list[str]:
    matches: list[str] = []
    for line in text.splitlines():
        if not atom_record(line):
            continue
        key = component_key(line)
        if (
            key.chain == selector["chain"]
            and key.resname == selector["resname"]
            and key.resid == selector["resid"]
            and key.icode == (selector.get("icode") or "")
            and line[12:16].strip() == selector["atom_name"]
        ):
            matches.append(line)
    return matches


def _residue_exists(text: str, selector: dict[str, Any]) -> bool:
    for line in text.splitlines():
        if not atom_record(line):
            continue
        key = component_key(line)
        if (
            key.chain == selector["chain"]
            and key.resname == selector["resname"]
            and key.resid == selector["resid"]
            and key.icode == (selector.get("icode") or "")
        ):
            return True
    return False


def _print_coordination_candidates(
    text: str,
    ion_line: str,
    *,
    cutoff_angstrom: float,
) -> None:
    """Print nearby atoms for review; never convert candidates into bonds."""
    candidates: list[tuple[float, str, str]] = []
    ion_serial = ion_line[6:11].strip()
    for line in text.splitlines():
        if not atom_record(line) or line[6:11].strip() == ion_serial:
            continue
        if element(line) == "H":
            continue
        distance = float(np.linalg.norm(xyz(line) - xyz(ion_line)))
        if distance <= cutoff_angstrom:
            key = component_key(line)
            selector = {
                "chain": key.chain,
                "resname": key.resname,
                "resid": key.resid,
                "icode": key.icode or None,
                "atom_name": line[12:16].strip(),
            }
            candidates.append((distance, _selector_label(selector), element(line)))
    print(
        f"Atoms within {cutoff_angstrom:.2f} A are candidates for review only; "
        "no bond is selected automatically:"
    )
    for distance, label, atom_element in sorted(candidates):
        print(f"  {label:24s} element={atom_element:>2s} distance={distance:6.3f} A")
    if not candidates:
        print("  none")


def _prompt_mcpb_coordinators(
    text: str,
    ion_selector: dict[str, Any],
) -> list[dict[str, Any]]:
    while True:
        raw = input(
            "Explicit MCPB coordinator atoms, comma separated "
            "(chain:RES:resid:ATOM): "
        ).strip()
        try:
            selectors = [
                _atom_selector_from_label(item.strip())
                for item in raw.split(",")
                if item.strip()
            ]
            if not selectors:
                raise ValueError("At least one explicit MCPB coordinator is required")
            labels = [_selector_label(item) for item in selectors]
            if len(labels) != len(set(labels)):
                raise ValueError("Each MCPB coordinator atom must be listed only once")
            for selector in selectors:
                matches = _matching_atom_lines(text, selector)
                if len(matches) != 1:
                    raise ValueError(
                        f"{_selector_label(selector)} matched {len(matches)} atoms; "
                        "exactly one is required"
                    )
                if element(matches[0]) == "H":
                    raise ValueError("MCPB coordinator selections must be heavy atoms")
                if selector == ion_selector:
                    raise ValueError("The metal ion cannot coordinate itself")
        except ValueError as error:
            print(error)
            continue
        print("Selected MCPB bonds:")
        for label in labels:
            print(f"  {_selector_label(ion_selector)} -- {label}")
        if _yes("Accept exactly these MCPB bonds"):
            return selectors


def _prompt_additional_mcpb_residues(text: str) -> list[dict[str, Any]]:
    while True:
        raw = input(
            "Additional complete MCPB residues, comma separated "
            "(chain:RES:resid), or none: "
        ).strip()
        if raw.lower() in {"", "none"}:
            return []
        try:
            selectors = [
                _residue_selector_from_label(item.strip())
                for item in raw.split(",")
                if item.strip()
            ]
            labels = [_selector_label(item, include_atom=False) for item in selectors]
            if len(labels) != len(set(labels)):
                raise ValueError("Each additional MCPB residue must be listed once")
            missing = [
                label
                for label, selector in zip(labels, selectors, strict=True)
                if not _residue_exists(text, selector)
            ]
            if missing:
                raise ValueError(f"Additional MCPB residues were not found: {missing}")
        except ValueError as error:
            print(error)
            continue
        return selectors


def _prompt_qm_component_keys(
    text: str,
    *,
    metal_key: ComponentKey,
) -> list[ComponentKey]:
    grouped = group_components(text)
    available = {
        key
        for key, lines in grouped.items()
        if key.resname not in WATERS and key != metal_key and not is_metal(lines)
    }
    print("Available non-protein QM components:")
    for key in sorted(available):
        print(f"  {key.label}")
    while True:
        raw = input(
            "QM ligand/cofactor components, comma separated (chain:RES:resid), "
            "or none: "
        ).strip()
        if raw.lower() in {"", "none"}:
            return []
        try:
            selected = parse_component_labels(raw)
            missing = selected - available
            if missing:
                raise ValueError(
                    "QM components were not found among the selected non-protein "
                    f"molecules: {sorted(item.label for item in missing)}"
                )
        except ValueError as error:
            print(error)
            continue
        print("Selected non-protein QM components:")
        for key in sorted(selected):
            print(f"  {key.label}")
        if _yes("Accept exactly these QM components"):
            return sorted(selected)


def _smiles_formal_charge(smiles: str) -> int:
    """Read the explicit molecular charge encoded by a SMILES string."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse that SMILES string")
    return int(Chem.GetFormalCharge(molecule))


def _retry_hydrogen_template(message: str) -> None:
    """Explain a recoverable template problem and let the user retry or abort."""
    print(f"\n{message}")
    if _choice("Hydrogen-template action", ("retry", "abort")) == "abort":
        raise ValueError(message)


def _rdkit_hydrogenate(
    key: ComponentKey,
    lines: list[str],
    *,
    mode: str,
    smiles: str,
) -> tuple[list[str], int]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    pdb = Chem.MolFromPDBBlock(
        "\n".join(lines) + "\nEND\n",
        sanitize=False,
        removeHs=False,
        proximityBonding=True,
    )
    if pdb is None:
        raise ValueError(f"RDKit could not read {key.label}")
    pdb_heavy = Chem.RemoveHs(pdb, sanitize=False)
    if mode == "ccd":
        response = requests.get(
            f"https://files.rcsb.org/ligands/download/{key.resname}_ideal.sdf",
            timeout=60,
        )
        response.raise_for_status()
        template = Chem.MolFromMolBlock(response.text, removeHs=False, sanitize=True)
    else:
        template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"No valid chemical template was obtained for {key.label}")
    template_heavy = Chem.RemoveHs(template)
    if template_heavy.GetNumAtoms() != pdb_heavy.GetNumAtoms():
        raise ValueError(
            f"{key.label}: template has {template_heavy.GetNumAtoms()} heavy atoms; "
            f"PDB has {pdb_heavy.GetNumAtoms()}"
        )
    assigned = AllChem.AssignBondOrdersFromTemplate(template_heavy, pdb_heavy)
    Chem.SanitizeMol(assigned)
    all_atom = Chem.AddHs(assigned, addCoords=True, addResidueInfo=True)
    Chem.SanitizeMol(all_atom)
    for index, atom in enumerate(all_atom.GetAtoms(), start=1):
        info = atom.GetPDBResidueInfo()
        if info is None:
            info = Chem.AtomPDBResidueInfo()
            info.SetName(f"H{index:<3}"[:4])
            atom.SetMonomerInfo(info)
        info.SetResidueName(key.resname)
        info.SetChainId(key.chain)
        info.SetResidueNumber(key.resid)
        info.SetInsertionCode(key.icode)
        info.SetIsHeteroAtom(True)
    generated = [
        line
        for line in Chem.MolToPDBBlock(all_atom).splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    original_heavy = [line for line in lines if element(line) != "H"]
    generated_heavy = [line for line in generated if element(line) != "H"]
    if len(original_heavy) != len(generated_heavy):
        raise ValueError(f"{key.label}: RDKit changed the heavy-atom count")
    for original, new in zip(original_heavy, generated_heavy, strict=True):
        if original[12:16].strip() != new[12:16].strip():
            raise ValueError(f"{key.label}: RDKit changed heavy-atom names/order")
        if np.linalg.norm(xyz(original) - xyz(new)) > 0.002:
            raise ValueError(f"{key.label}: RDKit changed a heavy-atom coordinate")
    return generated, int(Chem.GetFormalCharge(all_atom))


def _hydrogenate_waters(text: str) -> str:
    groups = group_components(text)
    solute = np.asarray(
        [
            xyz(line)
            for line in text.splitlines()
            if atom_record(line)
            and component_key(line).resname not in WATERS
            and element(line) != "H"
        ]
    )
    output: list[str] = []
    serial = sum(1 for line in text.splitlines() if atom_record(line)) + 1
    for line in text.splitlines():
        output.append(line)
        if not line.startswith("HETATM"):
            continue
        key = component_key(line)
        if key.resname not in WATERS or element(line) != "O":
            continue
        if any(element(item) == "H" for item in groups[key]):
            continue
        oxygen = xyz(line)
        nearest = solute[np.argmin(np.linalg.norm(solute - oxygen, axis=1))]
        bisector = oxygen - nearest
        bisector /= np.linalg.norm(bisector)
        trial = np.asarray([0.0, 0.0, 1.0])
        if abs(float(np.dot(bisector, trial))) > 0.9:
            trial = np.asarray([0.0, 1.0, 0.0])
        perpendicular = np.cross(bisector, trial)
        perpendicular /= np.linalg.norm(perpendicular)
        half_angle = np.deg2rad(104.52 / 2.0)
        for number, sign in ((1, 1.0), (2, -1.0)):
            vector = (
                np.cos(half_angle) * bisector
                + sign * np.sin(half_angle) * perpendicular
            )
            position = oxygen + 0.9572 * vector
            output.append(
                f"HETATM{serial:5d} {f'H{number}':>4s} {key.resname:>3s} "
                f"{key.chain:1s}{key.resid:4d}{key.icode:1s}   "
                f"{position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
                "  1.00  0.00           H  "
            )
            serial += 1
    return renumber_pdb("\n".join(output))


def collect_component_chemistry(
    cleaned_pdb: str,
    *,
    add_water_hydrogens: bool,
) -> tuple[str, dict[ComponentKey, dict[str, Any]]]:
    """Prompt for every chemistry-sensitive component field inside Colab."""
    groups = {
        key: lines
        for key, lines in group_components(cleaned_pdb).items()
        if key.resname not in WATERS
    }
    replacements: dict[ComponentKey, list[str]] = {}
    settings: dict[ComponentKey, dict[str, Any]] = {}
    for key, lines in sorted(groups.items()):
        print("\n" + "=" * 72)
        print(f"Component {key.label}: observed atoms {formula_from_lines(lines)}")
        if is_metal(lines):
            print(
                f"Detected one metal-like atom with element {element(lines[0]).capitalize()}."
            )
            if not _yes("Treat this component as a metal ion"):
                raise ValueError(
                    f"{key.label}: single-atom non-metal components are unsupported here"
                )
            charge = _required_int(
                "Formal metal ionic charge (positive oxidation state; Fe(II) is 2)",
                minimum=1,
            )
            multiplicity = _required_int("Metal multiplicity", minimum=1)
            atom_name = lines[0][12:16].strip()
            ion_selector = {
                "chain": key.chain,
                "resname": key.resname,
                "resid": key.resid,
                "icode": key.icode or None,
                "atom_name": atom_name,
            }
            metal_model = _choice("Metal model", ("nonbonded", "bonded_mcpb"))
            metal_settings: dict[str, Any] = {
                "role": "metal",
                "metal_model": metal_model,
                "charge": charge,
                "multiplicity": multiplicity,
                "element": element(lines[0]).capitalize(),
                "atom_name": atom_name,
            }
            if metal_model == "nonbonded":
                metal_settings["parameter_set"] = _choice(
                    "Amber nonbonded parameter set",
                    ("12_6_4", "12_6", "hfe", "iod", "cm"),
                )
            else:
                print(
                    "Bonded MCPB is an explicit single-site workflow. Nearby atoms are "
                    "shown only to help inspection; you must type every approved bond."
                )
                cutoff = _required_float("MCPB model cutoff in angstrom", minimum=0.1)
                _print_coordination_candidates(
                    cleaned_pdb,
                    lines[0],
                    cutoff_angstrom=cutoff,
                )
                coordinators = _prompt_mcpb_coordinators(
                    cleaned_pdb,
                    ion_selector,
                )
                additional_residues = _prompt_additional_mcpb_residues(cleaned_pdb)
                required_qm_keys: set[ComponentKey] = set()
                for selector in coordinators:
                    matched = _matching_atom_lines(cleaned_pdb, selector)[0]
                    candidate_key = component_key(matched)
                    if matched.startswith("HETATM") and candidate_key.resname not in WATERS:
                        required_qm_keys.add(candidate_key)
                for selector in additional_residues:
                    candidate_key = ComponentKey(
                        selector["chain"],
                        selector["resname"],
                        selector["resid"],
                        selector.get("icode") or "",
                    )
                    if candidate_key in group_components(cleaned_pdb):
                        required_qm_keys.add(candidate_key)
                while True:
                    qm_component_keys = _prompt_qm_component_keys(
                        cleaned_pdb,
                        metal_key=key,
                    )
                    missing_qm = required_qm_keys - set(qm_component_keys)
                    if not missing_qm:
                        break
                    print(
                        "Every coordinating/additional non-protein MCPB residue must be "
                        "selected as a QM component. Missing: "
                        + ", ".join(sorted(item.label for item in missing_qm))
                    )
                hessian_backend = _choice(
                    "MCPB Hessian method",
                    ("gxtb", "b3lyp_6-31g*"),
                )
                pyscf_threads = _required_int("PySCF CPU threads", minimum=1)
                pyscf_memory_mb = _required_int(
                    "PySCF maximum memory in MB", minimum=256
                )
                scf_algorithm = _choice(
                    "PySCF SCF algorithm",
                    ("diis", "newton", "adiis_then_diis", "adiis_then_newton"),
                )
                movable_atoms = _choice(
                    "GFN2 QM/MM movable atoms",
                    ("active_region_hydrogens", "all_active_region"),
                )
                allow_unusual_links = _yes(
                    "Authorize unusual peptide C-N QM/MM link boundaries after review"
                )
                metal_settings.update(
                    {
                        "provisional_parameter_set": _choice(
                            "Provisional Amber metal parameter set",
                            ("12_6_4", "12_6", "hfe", "iod", "cm"),
                        ),
                        "cutoff_angstrom": cutoff,
                        "coordinators": coordinators,
                        "additional_residues": additional_residues,
                        "qm_component_keys": qm_component_keys,
                        "hessian_backend": hessian_backend,
                        "pyscf_threads": pyscf_threads,
                        "pyscf_memory_mb": pyscf_memory_mb,
                        "scf_algorithm": scf_algorithm,
                        "movable_atoms": movable_atoms,
                        "allow_unusual_links": allow_unusual_links,
                        "charge_restraint": _choice(
                            "MCPB RESP charge restraint",
                            (
                                "backbone_heavy",
                                "all_ligating",
                                "backbone_all",
                                "backbone_and_cb",
                            ),
                        ),
                    }
                )
            settings[key] = metal_settings
            replacements[key] = lines
            continue

        role = _choice("Component role", ("ligand", "cofactor"))
        charge = _required_int("Net charge")
        multiplicity = _required_int("Multiplicity", minimum=1)
        while True:
            hydrogenation = _choice("Hydrogen source", ("ccd", "smiles", "keep"))
            smiles = ""
            if hydrogenation == "smiles":
                smiles = input(
                    "Paste the reviewed charged/protonated SMILES "
                    "(write formal charges as [O-], [N+], etc.): "
                ).strip()
                if not smiles:
                    _retry_hydrogen_template(f"{key.label}: SMILES cannot be empty")
                    continue
                try:
                    smiles_charge = _smiles_formal_charge(smiles)
                except ValueError as error:
                    _retry_hydrogen_template(f"{key.label}: {error}")
                    continue
                print(f"RDKit reads the SMILES net formal charge as {smiles_charge}.")
                if smiles_charge != charge:
                    guidance = (
                        f"{key.label}: the requested charge is {charge}, but this SMILES "
                        f"encodes charge {smiles_charge}. SMILES controls protonation: "
                        "for example, C(=O)O is a neutral carboxylic acid whereas "
                        "C(=O)[O-] is a deprotonated carboxylate."
                    )
                    if key.resname == "AKG" and charge == -2:
                        guidance += (
                            " If the intended species is the AKG dianion with the "
                            "connectivity shown here, review "
                            "O=C([O-])C(=O)CCC(=O)[O-]."
                        )
                    _retry_hydrogen_template(guidance)
                    continue
            if hydrogenation == "keep":
                if not any(element(line) == "H" for line in lines):
                    _retry_hydrogen_template(
                        f"{key.label}: the PDB contains no explicit hydrogens; "
                        "choose CCD or a reviewed SMILES"
                    )
                    continue
                generated = lines
                break
            print(
                "Mapping the template onto the crystallographic atom order. RDKit may "
                "report more than one matching pattern for symmetric or "
                "resonance-equivalent atoms; that warning alone is not a failure. "
                "Atom count, names, order, coordinates, and total charge are checked "
                "below, and the resulting structure must still be reviewed in 3D."
            )
            try:
                generated, template_charge = _rdkit_hydrogenate(
                    key, lines, mode=hydrogenation, smiles=smiles
                )
            except ValueError as error:
                _retry_hydrogen_template(f"{key.label}: {error}")
                continue
            print(
                f"RDKit reads the {hydrogenation.upper()} template net formal charge "
                f"as {template_charge}."
            )
            if template_charge != charge:
                template_guidance = (
                    f"{key.label}: the {hydrogenation.upper()} template charge "
                    f"{template_charge} does not match the requested charge {charge}. "
                    "Choose a reviewed charged SMILES or an explicitly protonated PDB."
                )
                if (
                    hydrogenation == "ccd"
                    and key.resname == "AKG"
                    and charge == -2
                    and template_charge == 0
                ):
                    template_guidance += (
                        " The downloaded RCSB AKG CCD entry is the neutral acid. "
                        "For the intended dianion, choose SMILES and review "
                        "O=C([O-])C(=O)CCC(=O)[O-]."
                    )
                _retry_hydrogen_template(template_guidance)
                continue
            break
        formula = formula_from_lines(generated)
        print(f"Generated exact formula: {formula}")
        if not _yes("Accept this component formula and hydrogenation"):
            raise ValueError(f"{key.label}: chemistry was not approved")
        settings[key] = {
            "role": role,
            "charge": charge,
            "multiplicity": multiplicity,
            "formula": formula,
            "atom_types": _choice("GAFF atom types", ("gaff2", "gaff")),
            "charge_method": _choice(
                "Charge method",
                (
                    "am1bcc",
                    "gas_resp_pyscf",
                    "qmmesp_pyscf",
                    "mcpb_resp_pyscf",
                ),
            ),
        }
        replacements[key] = generated

    bonded_metals = [
        (key, item)
        for key, item in settings.items()
        if item["role"] == "metal" and item["metal_model"] == "bonded_mcpb"
    ]
    if len(bonded_metals) > 1:
        raise ValueError(
            "mdprep v0.2 supports one independent bonded MCPB site per manifest"
        )
    if bonded_metals:
        protein_chains = {
            line[21:22].strip()
            for line in cleaned_pdb.splitlines()
            if line.startswith("ATOM  ")
        }
        if len(protein_chains) != 1:
            raise ValueError(
                "mdprep v0.2 bonded MCPB preparation requires a single protein chain; "
                f"the cleaned model contains {sorted(protein_chains)}"
            )
        _, bonded_settings = bonded_metals[0]
        site_residues = {
            ComponentKey(
                selector["chain"],
                selector["resname"],
                selector["resid"],
                selector.get("icode") or "",
            )
            for selector in (
                bonded_settings["coordinators"]
                + bonded_settings["additional_residues"]
            )
        }
        wrong_charge_method = sorted(
            key.label
            for key in site_residues
            if key in settings
            and settings[key]["role"] != "metal"
            and settings[key]["charge_method"] != "mcpb_resp_pyscf"
        )
        if wrong_charge_method:
            raise ValueError(
                "Coordinating/additional non-protein MCPB residues must use "
                "charge_method mcpb_resp_pyscf so the joint metal-site RESP charges "
                "replace provisional AM1-BCC charges. Rerun this cell for: "
                + ", ".join(wrong_charge_method)
            )

    output: list[str] = []
    emitted: set[ComponentKey] = set()
    for line in cleaned_pdb.splitlines():
        if line.startswith("HETATM"):
            key = component_key(line)
            if key in replacements:
                if key not in emitted:
                    output.extend(replacements[key])
                    emitted.add(key)
                continue
        if atom_record(line):
            output.append(line)
    all_atom = renumber_pdb("\n".join(output))
    if add_water_hydrogens:
        all_atom = _hydrogenate_waters(all_atom)
    return all_atom, settings


# ---- Manifest generation -------------------------------------------------


def _qmmesp(charge: int, multiplicity: int, *, embedded: bool) -> dict[str, Any]:
    return {
        "qm_engine": "pyscf",
        "method": "HF",
        "basis": "6-31G*",
        "embedding_cutoff_angstrom": None,
        "scf_charge": charge,
        "scf_spin": multiplicity - 1,
        "max_cycle": 100,
        "conv_tol": 1.0e-9,
        "num_threads": 2,
        "max_memory_mb": 10000,
        "grid": {
            "type": "merz_kollman",
            "vdw_scale_factors": [1.4, 1.6, 1.8, 2.0],
            "point_density_per_square_angstrom": 1.0,
            "exclude_inside_vdw_scale": 1.4,
            "max_points": 99999,
        },
        "resp_fitting": {"backend": "ambertools", "stage_2": True},
        "environment": {
            "include_protein": embedded,
            "include_waters": embedded,
            "include_other_ligands": embedded,
            "exclude_self_ligand": True,
        },
    }


def parse_overrides(value: str) -> list[dict[str, Any]]:
    """Parse comma-separated ``chain:RES:resid=STATE`` protonation overrides."""
    overrides: list[dict[str, Any]] = []
    for entry in value.split(","):
        if not entry.strip():
            continue
        match = re.fullmatch(
            r"([^:]+):([^:]+):(-?\d+)([A-Za-z]?)=([A-Za-z0-9+-]+)",
            entry.strip(),
        )
        if match is None:
            raise ValueError(
                "Each AA override must use chain:RES:resid=STATE; separate "
                "multiple overrides with commas"
            )
        chain, resname, resid, icode, state = match.groups()
        overrides.append(
            {
                "selector": {
                    "chain": "" if chain == "_" else chain,
                    "resname": resname.upper(),
                    "resid": int(resid),
                    "icode": icode or None,
                },
                "state": state.upper(),
                "reason": "Explicitly selected in the mdprep Colab workflow",
            }
        )
    return overrides


_SIDECHAIN_FORMAL_CHARGES = {
    "ARG": 1,
    "ASP": -1,
    "ASH": 0,
    "GLU": -1,
    "GLH": 0,
    "HIS": 0,
    "HID": 0,
    "HIE": 0,
    "HIP": 1,
    "LYS": 1,
    "LYN": 0,
    "CYS": 0,
    "CYM": -1,
    "CYX": 0,
}
_TITRATABLE_PROTEIN_RESIDUES = {
    "ARG",
    "ASP",
    "ASH",
    "GLU",
    "GLH",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "LYS",
    "LYN",
    "CYS",
    "CYM",
    "CYX",
}


def _selector_component_key(selector: dict[str, Any]) -> ComponentKey:
    return ComponentKey(
        selector["chain"],
        selector["resname"],
        selector["resid"],
        selector.get("icode") or "",
    )


def _override_states(
    overrides: list[dict[str, Any]],
) -> dict[ComponentKey, str]:
    states: dict[ComponentKey, str] = {}
    for item in overrides:
        key = _selector_component_key(item["selector"])
        if key in states:
            raise ValueError(f"Duplicate amino-acid protonation override: {key.label}")
        states[key] = item["state"]
    return states


def _single_open_shell_multiplicity(
    components: list[tuple[str, int]],
    *,
    region_name: str,
) -> int:
    open_shell = [(label, value) for label, value in components if value > 1]
    if len(open_shell) > 1:
        detail = ", ".join(f"{label}={value}" for label, value in open_shell)
        raise ValueError(
            f"Cannot derive the {region_name} multiplicity because it contains "
            f"multiple open-shell components ({detail}) and their spin coupling is "
            "not determined by isolated-component multiplicities. This Colab will not "
            "guess a coupled spin state."
        )
    return open_shell[0][1] if open_shell else 1


def _derive_bonded_electronic_state(
    metal_key: ComponentKey,
    metal: dict[str, Any],
    chemistry: dict[ComponentKey, dict[str, Any]],
    overrides: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive MCPB/QM charges and spins from reviewed component chemistry."""
    override_by_key = _override_states(overrides)
    coordinator_by_key = {
        _selector_component_key(selector): selector
        for selector in metal["coordinators"]
    }
    site_keys = {
        *coordinator_by_key,
        *(
            _selector_component_key(selector)
            for selector in metal["additional_residues"]
        ),
    }
    charge_terms: list[tuple[str, int]] = [(metal_key.label, metal["charge"])]
    model_multiplicities: list[tuple[str, int]] = [
        (metal_key.label, metal["multiplicity"])
    ]
    for key in sorted(site_keys):
        component = chemistry.get(key)
        if component is not None:
            if component["role"] == "metal":
                raise ValueError(
                    "A second metal cannot be hidden in a bonded MCPB coordinator list"
                )
            charge_terms.append((key.label, component["charge"]))
            model_multiplicities.append((key.label, component["multiplicity"]))
            continue
        if key.resname in WATERS:
            charge_terms.append((key.label, 0))
            model_multiplicities.append((key.label, 1))
            continue
        state = override_by_key.get(key)
        coordinator = coordinator_by_key.get(key)
        if key.resname in {"HIS", "HID", "HIE", "HIP"} and coordinator:
            donor = coordinator["atom_name"].upper()
            if donor not in {"ND1", "NE2"}:
                raise ValueError(
                    f"Cannot derive protonation for {key.label} coordinated through "
                    f"{donor}; use an explicit AA override"
                )
            required_state = "HIE" if donor == "ND1" else "HID"
            if state is not None and state != required_state:
                raise ValueError(
                    f"The explicit {state} override for {key.label} conflicts with "
                    f"coordination through {donor}, which requires {required_state}"
                )
            state = required_state
        if state is None and key.resname in _TITRATABLE_PROTEIN_RESIDUES:
            raise ValueError(
                f"Cannot derive the bonded-site charge until {key.label} has an "
                "explicit protonation state. Add it to AA_OVERRIDES using, for "
                f"example, {key.label}={key.resname}."
            )
        state = state or key.resname
        if state not in _SIDECHAIN_FORMAL_CHARGES:
            charge = 0
        else:
            charge = _SIDECHAIN_FORMAL_CHARGES[state]
        charge_terms.append((f"{key.label}({state})", charge))
        model_multiplicities.append((key.label, 1))

    model_charge = sum(value for _, value in charge_terms)
    model_multiplicity = _single_open_shell_multiplicity(
        model_multiplicities,
        region_name="MCPB model",
    )
    qm_multiplicities = [(metal_key.label, metal["multiplicity"])]
    qm_multiplicities.extend(
        (key.label, chemistry[key]["multiplicity"])
        for key in metal["qm_component_keys"]
    )
    qm_charge_terms = list(charge_terms)
    qm_charge_terms.extend(
        (key.label, chemistry[key]["charge"])
        for key in metal["qm_component_keys"]
        if key not in site_keys
    )
    total_qm_multiplicity = _single_open_shell_multiplicity(
        qm_multiplicities,
        region_name="complete QM region",
    )
    return {
        "small_model_charge": model_charge,
        "small_model_spin": model_multiplicity,
        "large_model_charge": model_charge,
        "large_model_spin": model_multiplicity,
        "total_qm_charge": sum(value for _, value in qm_charge_terms),
        "total_qm_multiplicity": total_qm_multiplicity,
        "charge_terms": charge_terms,
        "qm_charge_terms": qm_charge_terms,
        "model_multiplicities": model_multiplicities,
        "qm_multiplicities": qm_multiplicities,
    }


def _print_derived_electronic_state(
    metal_key: ComponentKey,
    state: dict[str, Any],
) -> None:
    print(f"\nAutomatically derived electronic state for {metal_key.label}:")
    print(
        "  MCPB charge = "
        + " + ".join(f"{label}({charge:+d})" for label, charge in state["charge_terms"])
        + f" = {state['small_model_charge']:+d}"
    )
    print(
        f"  MCPB small/large multiplicity = {state['small_model_spin']} "
        "(from the reviewed component multiplicities)"
    )
    print(
        "  Complete QM-region charge = "
        + " + ".join(
            f"{label}({charge:+d})" for label, charge in state["qm_charge_terms"]
        )
        + f" = {state['total_qm_charge']:+d}"
    )
    print(
        f"  Complete QM-region multiplicity = {state['total_qm_multiplicity']} "
        "(from the reviewed QM-component multiplicities)"
    )


def _component_id(key: ComponentKey, item: dict[str, Any]) -> str:
    return f"{item['role']}_{key.chain or 'blank'}_{key.resname}_{key.resid}"


def _metal_site_id(key: ComponentKey) -> str:
    return f"metal_{key.chain or 'blank'}_{key.resid}"


def _mcpb_pyscf_settings(item: dict[str, Any]) -> dict[str, Any]:
    """Build one reviewed fixed-geometry Hessian/large-model ESP protocol."""
    settings: dict[str, Any] = {
        "geometry_source": "qmmm_refinement",
        "hessian_backend": (
            "xtb" if item["hessian_backend"] == "gxtb" else "pyscf"
        ),
        "method": "B3LYP",
        "basis": "6-31G*",
        "max_cycle": 150,
        "conv_tol": 1.0e-9,
        "scf_algorithm": item["scf_algorithm"],
        "initial_guess": "minao",
        "adiis_precondition_cycles": 15,
        "level_shift_mode": "static",
        "level_shift_hartree": 0.0,
        "damping_factor": 0.0,
        "diis_space": 8,
        "large_model_density_fitting": False,
        "dft_grid_level": 3,
        "num_threads": item["pyscf_threads"],
        "max_memory_mb": item["pyscf_memory_mb"],
        "embedding_cutoff_angstrom": None,
        "embedding_min_distance_angstrom": 1.2,
        "esp_batch_size": 256,
    }
    if item["hessian_backend"] == "gxtb":
        settings["xtb"] = {
            "model": "gxtb",
            "executable": "/content/gxtb-v2.0.1/xtb-6.7.1/bin/xtb",
            "release_tag": "v2.0.1",
            "expected_executable_sha256": (
                "1b4e30b68ed4e88b4075f60d97f4756e"
                "e440fe92d3294cc20826d53f8121cd26"
            ),
            "accuracy": 0.001,
            "num_threads": 1,
            "max_scf_iterations": 1000,
        }
    return settings


def build_manifest(
    chemistry: dict[ComponentKey, dict[str, Any]],
    options: dict[str, Any],
) -> dict[str, Any]:
    protonation_overrides = parse_overrides(options["aa_overrides"])
    bonded_electronic_states: dict[ComponentKey, dict[str, Any]] = {}
    for key, item in chemistry.items():
        if item["role"] == "metal" and item["metal_model"] == "bonded_mcpb":
            state = _derive_bonded_electronic_state(
                key,
                item,
                chemistry,
                protonation_overrides,
            )
            bonded_electronic_states[key] = state
            _print_derived_electronic_state(key, state)
    ligands: list[dict[str, Any]] = []
    metals: list[dict[str, Any]] = []
    for key, item in chemistry.items():
        if item["role"] == "metal":
            ion_selector = {
                "chain": key.chain,
                "resname": key.resname,
                "resid": key.resid,
                "icode": key.icode or None,
                "atom_name": item["atom_name"],
            }
            site: dict[str, Any] = {
                "id": _metal_site_id(key),
                "model": item["metal_model"],
                "ions": [
                    {
                        "selector": ion_selector,
                        "element": item["element"],
                        "charge": item["charge"],
                        "multiplicity": item["multiplicity"],
                    }
                ],
            }
            if item["metal_model"] == "nonbonded":
                site["nonbonded"] = {"parameter_set": item["parameter_set"]}
                site["mcpb"] = None
            else:
                electronic_state = bonded_electronic_states[key]
                site["nonbonded"] = None
                site["mcpb"] = {
                    "executable": "MCPB.py",
                    "workflow": "pyscf",
                    "provisional_nonbonded_parameter_set": item[
                        "provisional_parameter_set"
                    ],
                    "cutoff_angstrom": item["cutoff_angstrom"],
                    "bonds": [
                        {"ion": ion_selector, "coordinator": coordinator}
                        for coordinator in item["coordinators"]
                    ],
                    "additional_residues": item["additional_residues"],
                    "force_constant_method": "seminario",
                    "charge_restraint": item["charge_restraint"],
                    "software_version": "gau",
                    "small_model_charge": electronic_state["small_model_charge"],
                    "small_model_spin": electronic_state["small_model_spin"],
                    "large_model_charge": electronic_state["large_model_charge"],
                    "large_model_spin": electronic_state["large_model_spin"],
                    "scale_factor": 1.0,
                    "large_opt": 0,
                    "artifacts": None,
                    "pyscf": _mcpb_pyscf_settings(item),
                    "parameter_comparison": None,
                }
            metals.append(site)
            continue
        method = item["charge_method"]
        ligands.append(
            {
                "id": _component_id(key, item),
                "selector": {
                    "chain": key.chain,
                    "resname": key.resname,
                    "resid": key.resid,
                    "icode": key.icode or None,
                },
                "net_charge": item["charge"],
                "multiplicity": item["multiplicity"],
                "expected_formula": item["formula"],
                "atom_types": item["atom_types"],
                "charge_method": method,
                "user_mol2": None,
                "user_frcmod": None,
                "preserve_atom_names": True,
                "preserve_coordinates": True,
                "allow_atom_renaming": False,
                "allow_coordinate_changes": False,
                "qmmesp": (
                    _qmmesp(
                        item["charge"],
                        item["multiplicity"],
                        embedded=method == "qmmesp_pyscf",
                    )
                    if method in {"gas_resp_pyscf", "qmmesp_pyscf"}
                    else None
                ),
            }
        )
    md_enabled = options["md_mode"] != "none"
    steps = (
        1 if options["md_mode"] == "equilibration_only" else options["production_steps"]
    )
    md: dict[str, Any] = {"enabled": False}
    if md_enabled:
        md = {
            "enabled": True,
            "protocol": "roe_brooks_2020",
            "temperature_kelvin": 300.0,
            "pressure_atmosphere": 1.0,
            "platform": options["md_platform"],
            "device_index": 0 if options["md_platform"] == "CUDA" else None,
            "precision": "mixed",
            "cpu_threads": None,
            "random_seed": 20260817,
            "step08_ensemble": "NPT",
            "density": {
                "increment_ns": 1.0,
                "report_interval_ps": 1.0,
                "window_ps": 300.0,
                "slope_threshold_g_ml_ps": 1.0e-6,
                "mean_difference_threshold_g_ml": 0.02,
                "minimum_points": 50,
                "maximum_duration_ns": 10.0,
            },
            "production": {
                "steps": steps,
                "timestep_fs": 2.0,
                "trajectory_interval_steps": min(10000, steps),
                "state_interval_steps": min(5000, steps),
                "checkpoint_interval_steps": min(50000, steps),
            },
        }
    bonded_sites = [
        (key, item)
        for key, item in chemistry.items()
        if item["role"] == "metal" and item["metal_model"] == "bonded_mcpb"
    ]
    if len(bonded_sites) > 1:
        raise ValueError("mdprep v0.2 supports one bonded MCPB site per manifest")
    refinement: dict[str, Any] = {"enabled": False}
    if bonded_sites:
        metal_key, metal_item = bonded_sites[0]
        electronic_state = bonded_electronic_states[metal_key]
        qm_ligand_ids: list[str] = []
        for component_key_value in metal_item["qm_component_keys"]:
            component = chemistry.get(component_key_value)
            if component is None or component["role"] == "metal":
                raise ValueError(
                    f"Invalid QM ligand/cofactor selection: {component_key_value.label}"
                )
            qm_ligand_ids.append(_component_id(component_key_value, component))
        refinement = {
            "enabled": True,
            "backend": "ash",
            "qm_method": "gfn2_xtb",
            "embedding": "electrostatic",
            "qm_components": {
                "ligands": qm_ligand_ids,
                "metal_sites": [_metal_site_id(metal_key)],
                "metal_coordinating_residues": {},
            },
            "active_region_cutoff_angstrom": 4.0,
            "active_water_cutoff_angstrom": 8.0,
            "include_contact_waters": True,
            "movable_atoms": metal_item["movable_atoms"],
            "post_refinement_protonation": "reuse_initial",
            "total_qm_multiplicity": electronic_state["total_qm_multiplicity"],
            "ash": {
                "python_executable": "python",
                "xtb_executable": "xtb",
                "allow_unusual_link_boundaries": metal_item[
                    "allow_unusual_links"
                ],
                "max_iterations": 250,
                "num_cores": metal_item["pyscf_threads"],
                "platform": "CPU",
                "xtb_max_iterations": 500,
                "electronic_temperature_kelvin": 300.0,
                "accuracy": 0.1,
            },
            "mace_polar1": None,
        }
    manifest = {
        "project": {
            "name": options["project_name"],
            "input_structure": "prepared_input.pdb",
            "output_dir": "prepared/system",
        },
        "structure": {
            "keep_crystal_waters": True,
            "altloc_policy": "highest_occupancy",
            "remove_unknown_heterogens": False,
            "preserve_chain_ids": True,
            "remove_input_hydrogens": False,
        },
        "protein": {
            "forcefield": options["forcefield"],
            "water_model": options["water_model"],
        },
        "protonation": {
            "ph": options["ph"],
            "method": options["protonation_method"],
            "overrides": protonation_overrides,
            "histidine": {
                "neutral_tautomer_method": options["histidine_method"],
                "xtb": {
                    "executable": "xtb",
                    "model": "gfn2",
                    "mode": "opt",
                    "opt_level": "loose",
                    "solvent": "water",
                    "cutoff_angstrom": 5.0,
                    "extra_args": [],
                },
            },
        },
        "disulfides": {
            "auto_detect": True,
            "detection_cutoff_angstrom": 2.2,
            "force": [],
            "forbid": [],
        },
        "ligands": ligands,
        "metals": metals,
        "refinement": refinement,
        "solvation": {
            "enabled": options["solvate"],
            "box": "truncated_octahedron",
            "buffer_angstrom": options["buffer_angstrom"],
            "neutralize": True,
            "salt_concentration_molar": options["salt_molar"],
            "positive_ion": "Na+",
            "negative_ion": "Cl-",
        },
        "validation": {
            "run_openmm_energy_check": True,
            "fail_on_warnings": False,
            "fail_on_missing_parameters": True,
            "fail_on_noninteger_ligand_charge": True,
        },
        "molecular_dynamics": md,
    }
    from mdprep.config.models import ManifestConfig

    ManifestConfig.model_validate(manifest)
    return manifest


def write_bundle(manifest: dict[str, Any]) -> Path:
    manifest_path = WORKDIR / "system.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    bundle = WORKDIR / "mdprep_input_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "input_raw.pdb",
            "cleaned.pdb",
            "prepared_input.pdb",
            "system.yaml",
        ):
            path = WORKDIR / name
            if path.is_file():
                archive.write(path, arcname=name)
    return bundle
