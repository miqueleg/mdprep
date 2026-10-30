"""A noncanonical residue parameterised from its SMILES, for a protein chain.

The residue is parameterised between an acetyl and a methylamide
(``capping.py``), the way amino acids always have been, and built for the
protein force field it joins:

* AM1-BCC charges for the capped molecule (antechamber), with the backbone's
  then fixed to the force field's and the caps dropped (``charges.py``);
* GAFF2 atom types on the side chain, and the protein force field's on the
  backbone -- N, H, CA, HA, C and O, and CB with its hydrogens when it is an
  sp3 carbon. Every term about the backbone, its phi and psi torsions and
  the CA-CB bond included, is then the protein force field's own, and the
  peptide bonds to the neighbours are exactly those of any other residue;
* parmchk2's estimates for the terms joining the two, and nothing else
  (``frcmod.py``), with ff19SB's CMAP for its parent;
* a tleap library whose head is N and tail is C, so tleap bonds the residue
  into the chain as it does any amino acid.

Every command goes through mdprep.external.runner, and every file it reads or
writes stays in the residue's directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mdprep.ambertools.commands import AmberToolRun, AmberToolsError
from mdprep.ambertools.mol2 import Mol2Error, read_mol2
from mdprep.config.models import NoncanonicalResidueConfig
from mdprep.external.discovery import which_executable
from mdprep.external.runner import run_command
from mdprep.leap.forcefields import ResidueLibraryFiles, protein_leaprc
from mdprep.leap.log_parser import LeapLogError, assert_tleap_success
from mdprep.leap.runner import TLeapRun, TLeapRunError, run_tleap
from mdprep.residues.capping import CappedResidue, ChainContext, capped_residue
from mdprep.residues.charges import ResidueCharges, constrain_to_backbone
from mdprep.residues.frcmod import (
    attention_lines,
    cmap_section,
    ff19sb_frcmod_path,
    is_gaff_type,
    junction_parameters,
    parent_cmap,
)
from mdprep.residues.offlib import OffLibraryError, read_off_unit

# The protein force field's type for CA.
ALPHA_CARBON_TYPE = {"ff14SB": "CX", "ff19SB": "XC"}

# Backbone atoms of the residue and of the caps, by the protein's types.
BACKBONE_TYPES = {
    ("RES", "N"): "N",
    ("RES", "C"): "C",
    ("RES", "O"): "O",
    ("ACE", "CAY"): "CT",
    ("ACE", "HY1"): "HC",
    ("ACE", "HY2"): "HC",
    ("ACE", "HY3"): "HC",
    ("ACE", "CY"): "C",
    ("ACE", "OY"): "O",
    ("NME", "NT"): "N",
    ("NME", "HNT"): "H",
    ("NME", "CAT"): "CT",
    ("NME", "HT1"): "H1",
    ("NME", "HT2"): "H1",
    ("NME", "HT3"): "H1",
}

# Types parmchk2 does not know, and the one it is asked about instead.
PARMCHK2_TYPE = {"XC": "CX"}

# GAFF2 hydrogens on an sp3 carbon, and the protein force field's equivalent.
HYDROGEN_TYPES = {"hc": "HC", "h1": "H1", "h2": "H2", "h3": "H3"}


class ResidueParameterisationError(ValueError):
    """Raised when a residue cannot be parameterised safely."""


@dataclass
class ParameterisedResidue:
    resname: str
    parameterised_from: str
    smiles: str
    smiles_form: str
    net_charge: int
    parent: str | None
    protein_forcefield: str
    files: ResidueLibraryFiles
    capped_input_mol2_path: Path
    # antechamber's typing of the capped molecule, GAFF2 throughout: what a
    # RESP fit of it reads the topology from.
    antechamber_mol2_path: Path
    capped_mol2_path: Path
    residue_mol2_path: Path
    raw_frcmod_path: Path
    charges: ResidueCharges
    atom_types: dict[str, str]
    am1bcc_charges: dict[str, float]
    antechamber: AmberToolRun
    parmchk2: AmberToolRun
    tleap: TLeapRun
    cmap: str | None
    dropped_protein_terms: list[str]
    attention: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "resname": self.resname,
            "parameterised_from": self.parameterised_from,
            "smiles": self.smiles,
            "smiles_form": self.smiles_form,
            "net_charge": self.net_charge,
            "parent": self.parent,
            "protein_forcefield": self.protein_forcefield,
            "library_path": str(self.files.library_path),
            "frcmod_path": str(self.files.frcmod_path),
            "capped_input_mol2_path": str(self.capped_input_mol2_path),
            "antechamber_mol2_path": str(self.antechamber_mol2_path),
            "capped_mol2_path": str(self.capped_mol2_path),
            "residue_mol2_path": str(self.residue_mol2_path),
            "raw_frcmod_path": str(self.raw_frcmod_path),
            "atom_types": dict(self.atom_types),
            "am1bcc_charges": dict(self.am1bcc_charges),
            "charges": self.charges.to_dict(),
            "cmap": self.cmap,
            "protein_terms_left_to_the_force_field": len(self.dropped_protein_terms),
            "parameters_parmchk2_could_not_estimate": list(self.attention),
            "antechamber": self.antechamber.to_dict(),
            "parmchk2": self.parmchk2.to_dict(),
            "tleap": self.tleap.to_dict(),
            "warnings": list(self.warnings),
        }


def parameterise_residue(
    block: NoncanonicalResidueConfig,
    context: ChainContext,
    *,
    protein_forcefield: str,
    work_dir: str | Path,
    amberhome: Path | None = None,
) -> ParameterisedResidue:
    """Parameterise ``block``'s residues from the first one, in ``context``."""

    assert block.smiles is not None and block.net_charge is not None
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    name = block.resname
    label = f"noncanonical residue {name} ({context.residue.id.display()})"
    capped = capped_residue(context, block.smiles, net_charge=block.net_charge, label=label)

    capped_input = work / f"{name}.capped.input.mol2"
    write_tripos_mol2(capped_input, capped, resname=name)
    antechamber_mol2 = work / f"{name}.capped.antechamber.mol2"
    antechamber = _run_tool(
        "antechamber",
        [
            "-i", capped_input.name, "-fi", "mol2",
            "-o", antechamber_mol2.name, "-fo", "mol2",
            "-c", "bcc", "-nc", str(block.net_charge), "-m", "1",
            "-at", block.atom_types, "-rn", name,
            "-an", "n", "-du", "n", "-seq", "n", "-pf", "y", "-s", "2",
        ],
        work_dir=work,
        output=antechamber_mol2,
    )
    try:
        typed = read_mol2(antechamber_mol2)
    except Mol2Error as exc:
        raise ResidueParameterisationError(f"{label}: {exc}") from exc
    expected = [atom.name for atom in capped.atoms]
    if [atom.name for atom in typed.atoms] != expected:
        raise ResidueParameterisationError(
            f"{label}: antechamber changed the capped residue's atoms from {expected} "
            f"to {[atom.name for atom in typed.atoms]}."
        )
    gaff_types = [atom.atom_type for atom in typed.atoms]
    fitted = [
        (atom.role, atom.name, typed_atom.charge)
        for atom, typed_atom in zip(capped.atoms, typed.atoms)
    ]

    types = protein_backbone_types(capped, gaff_types, protein_forcefield=protein_forcefield)
    charges = constrain_to_backbone(fitted, block.net_charge)
    warnings = list(charges.warnings)
    if any(
        atom.role == "RES" and atom.name == "CB" and is_gaff_type(types[index])
        for index, atom in enumerate(capped.atoms)
    ):
        warnings.append(
            f"{name}'s CB is not an sp3 carbon, so the terms about CA that involve it "
            "are parmchk2's estimates from GAFF2 rather than the protein force field's."
        )

    final_charges = [
        charges.charges[atom.name] if atom.role == "RES" else charge
        for atom, (_, _, charge) in zip(capped.atoms, fitted)
    ]
    capped_mol2 = work / f"{name}.capped.mol2"
    # parmchk2's correspondence table predates ff19SB's XC; it is asked about
    # CX, the same carbon in ff14SB, and its answers renamed back below.
    parmchk2_types = [PARMCHK2_TYPE.get(atom_type, atom_type) for atom_type in types]
    write_tripos_mol2(
        capped_mol2, capped, resname=name, types=parmchk2_types, charges=final_charges
    )
    raw_frcmod = work / f"{name}.parmchk2.frcmod"
    parmchk2 = _run_tool(
        "parmchk2",
        ["-i", capped_mol2.name, "-f", "mol2", "-o", raw_frcmod.name, "-s", block.atom_types],
        work_dir=work,
        output=raw_frcmod,
    )
    junction, dropped = junction_parameters(
        raw_frcmod.read_text(encoding="utf-8"),
        rename={value: key for key, value in PARMCHK2_TYPE.items() if key in types},
    )
    cmap_note: str | None = None
    if protein_forcefield == "ff19SB":
        if block.parent is None:
            raise ResidueParameterisationError(
                f"{label}: ff19SB needs the residue's parent, whose backbone CMAP it takes."
            )
        source = ff19sb_frcmod_path(amberhome or _amberhome())
        if not source.is_file():
            raise ResidueParameterisationError(f"{label}: {source} does not exist.")
        cmap = parent_cmap(source.read_text(encoding="utf-8"), block.parent)
        junction += cmap_section(name, block.parent, cmap, "ff19SB")
        cmap_note = f"the {cmap.title} ff19SB applies to {block.parent}"
    frcmod = work / f"{name}.frcmod"
    frcmod.write_text(junction, encoding="utf-8")
    attention = attention_lines(junction)
    library, residue_mol2, tleap = build_residue_library(
        capped,
        resname=name,
        types=types,
        charges=final_charges,
        protein_forcefield=protein_forcefield,
        atom_types=block.atom_types,
        frcmod=frcmod,
        label=label,
    )

    residue_types = {
        atom.name: types[index] for index, atom in enumerate(capped.atoms) if atom.role == "RES"
    }
    result = ParameterisedResidue(
        resname=name,
        parameterised_from=context.residue.id.display(),
        smiles=block.smiles,
        smiles_form=capped.smiles_form,
        net_charge=block.net_charge,
        parent=block.parent,
        protein_forcefield=protein_forcefield,
        files=ResidueLibraryFiles(resname=name, library_path=library, frcmod_path=frcmod),
        capped_input_mol2_path=capped_input,
        antechamber_mol2_path=antechamber_mol2,
        capped_mol2_path=capped_mol2,
        residue_mol2_path=residue_mol2,
        raw_frcmod_path=raw_frcmod,
        charges=charges,
        atom_types=residue_types,
        am1bcc_charges={name: charge for role, name, charge in fitted if role == "RES"},
        antechamber=antechamber,
        parmchk2=parmchk2,
        tleap=tleap,
        cmap=cmap_note,
        dropped_protein_terms=dropped,
        attention=attention,
        warnings=warnings,
    )
    (work / "parameters.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_residue_library(
    capped: CappedResidue,
    *,
    resname: str,
    types: list[str],
    charges: list[float],
    protein_forcefield: str,
    atom_types: str,
    frcmod: Path,
    label: str,
) -> tuple[Path, Path, TLeapRun]:
    """The residue's mol2 and its tleap library, beside ``frcmod``.

    ``types`` and ``charges`` are per atom of ``capped``; only the residue's
    own reach the files. The library is read back to check that every atom
    has its element: tleap takes one from the type, and has none for some.
    """

    work = frcmod.parent
    residue_mol2 = work / f"{resname}.mol2"
    write_tripos_mol2(
        residue_mol2, capped, resname=resname, types=types, charges=charges, roles=("RES",)
    )
    library = work / f"{resname}.lib"
    script = work / "tleap.library.in"
    elements = {
        atom.name: atom.element
        for index, atom in enumerate(capped.atoms)
        if atom.role == "RES" and not is_gaff_type(types[index])
    }
    script.write_text(
        library_script(
            resname=resname,
            protein_forcefield=protein_forcefield,
            atom_types=atom_types,
            mol2_name=residue_mol2.name,
            frcmod_name=frcmod.name,
            library_name=library.name,
            elements=elements,
        ),
        encoding="utf-8",
    )
    try:
        tleap = run_tleap(script, work_dir=work)
        assert_tleap_success(tleap.summary, fail_on_warnings=False, context=f"{resname} library")
    except (TLeapRunError, LeapLogError) as exc:
        raise ResidueParameterisationError(f"{label}: {exc}") from exc
    if not library.is_file() or library.stat().st_size == 0:
        raise ResidueParameterisationError(f"{label}: tleap did not write {library}.")
    written = library_elements(library.read_text(encoding="utf-8"), resname)
    wrong = sorted(
        atom.name
        for atom in capped.atoms
        if atom.role == "RES"
        and written.get(atom.name) != ATOMIC_NUMBERS.get(atom.element.capitalize())
    )
    if wrong:
        raise ResidueParameterisationError(
            f"{label}: tleap's library gives {', '.join(wrong)} the wrong element: "
            f"{ {atom: written.get(atom) for atom in wrong} }."
        )
    return library, residue_mol2, tleap


def protein_backbone_types(
    capped: CappedResidue, gaff_types: list[str], *, protein_forcefield: str
) -> list[str]:
    """Each atom's type: the protein force field's on the backbone, GAFF2's elsewhere.

    The backbone is the residue's N and its hydrogen, CA and its hydrogens, C
    and O, and the caps. An sp3
    carbon bonded to CA -- CB, or both of an Aib's -- becomes CT, and its
    hydrogens the protein's hydrogen types, so that every term centred on CA
    is one the protein force field already has.
    """

    types = list(gaff_types)
    index_of = {(atom.role, atom.name): index for index, atom in enumerate(capped.atoms)}
    for key, amber_type in BACKBONE_TYPES.items():
        if key in index_of:
            types[index_of[key]] = amber_type
    alpha = index_of[("RES", "CA")]
    types[alpha] = ALPHA_CARBON_TYPE[protein_forcefield]
    neighbours: dict[int, set[int]] = {}
    for first, second, _ in capped.bonds:
        neighbours.setdefault(first, set()).add(second)
        neighbours.setdefault(second, set()).add(first)
    nitrogen = index_of[("RES", "N")]
    backbone = {nitrogen, index_of[("RES", "C")]}
    # The amide hydrogen, and CA's own (HA, or a glycine-like HA2 and HA3).
    for hydrogen in neighbours[nitrogen]:
        if capped.atoms[hydrogen].element == "H":
            types[hydrogen] = "H"
    for hydrogen in neighbours[alpha]:
        if capped.atoms[hydrogen].element == "H":
            types[hydrogen] = "H1"
    for carbon in sorted(neighbours[alpha] - backbone):
        atom = capped.atoms[carbon]
        if atom.element != "C" or gaff_types[carbon] != "c3":
            continue
        types[carbon] = "CT"
        for hydrogen in neighbours[carbon]:
            if capped.atoms[hydrogen].element == "H" and gaff_types[hydrogen] in HYDROGEN_TYPES:
                types[hydrogen] = HYDROGEN_TYPES[gaff_types[hydrogen]]
    return types


def library_script(
    *,
    resname: str,
    protein_forcefield: str,
    atom_types: str,
    mol2_name: str,
    frcmod_name: str,
    library_name: str,
    elements: dict[str, str] | None = None,
) -> str:
    """tleap commands that turn the residue's mol2 into a library of one unit.

    Head N and tail C make tleap bond it to the residues before and after it
    when it reads a chain, as it does any amino acid's. ``elements`` are set
    explicitly: loadmol2 takes an element from the type, and has none for
    ff19SB's XC, which it would leave unknown and savepdb then write as
    calcium.
    """

    return "\n".join(
        [
            f"source {protein_leaprc(protein_forcefield)}",
            f"source leaprc.{atom_types}",
            f"loadamberparams {frcmod_name}",
            f"{resname} = loadmol2 {mol2_name}",
            *(
                f"set {resname}.1.{atom} element {element}"
                for atom, element in (elements or {}).items()
            ),
            f"set {resname} head {resname}.1.N",
            f"set {resname} tail {resname}.1.C",
            f"set {resname}.1 connect0 {resname}.1.N",
            f"set {resname}.1 connect1 {resname}.1.C",
            f"set {resname}.1 restype protein",
            f'set {resname} name "{resname}"',
            f"check {resname}",
            f"saveoff {resname} {library_name}",
            "quit",
            "",
        ]
    )


# The elements a residue can hold, by atomic number, as a tleap library stores them.
ATOMIC_NUMBERS = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16,
    "Cl": 17, "Se": 34, "Br": 35, "I": 53,
}


def library_elements(text: str, resname: str) -> dict[str, int]:
    """Atom name to atomic number in a tleap OFF library's unit ``resname``."""

    try:
        unit = read_off_unit(text, resname)
    except OffLibraryError as exc:
        raise ResidueParameterisationError(str(exc)) from exc
    return {atom.name: atom.atomic_number for atom in unit.atoms}


def write_tripos_mol2(
    path: Path,
    capped: CappedResidue,
    *,
    resname: str,
    types: list[str] | None = None,
    charges: list[float] | None = None,
    roles: tuple[str, ...] = ("ACE", "RES", "NME"),
) -> None:
    """The capped residue, or the atoms of ``roles``, as a Tripos mol2 with bonds.

    Without ``types`` the atoms are typed by element, for antechamber to type
    them itself from the bonds given here rather than from the coordinates.
    """

    chosen = [index for index, atom in enumerate(capped.atoms) if atom.role in roles]
    number = {index: serial for serial, index in enumerate(chosen, start=1)}
    bonds = [
        (number[first], number[second], order)
        for first, second, order in capped.bonds
        if first in number and second in number
    ]
    lines = [
        "@<TRIPOS>MOLECULE",
        resname,
        f"{len(chosen)} {len(bonds)} 1 0 0",
        "SMALL",
        "USER_CHARGES" if charges is not None else "NO_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for index in chosen:
        atom = capped.atoms[index]
        atom_type = types[index] if types is not None else atom.element
        charge = charges[index] if charges is not None else 0.0
        lines.append(
            f"{number[index]:7d} {atom.name:<4s} {atom.x:10.4f} {atom.y:10.4f} {atom.z:10.4f} "
            f"{atom_type:<5s} 1 {resname:<4s} {charge: .6f}"
        )
    lines.append("@<TRIPOS>BOND")
    for serial, (first, second, order) in enumerate(bonds, start=1):
        lines.append(f"{serial:6d} {first:5d} {second:5d} {order}")
    lines.append("@<TRIPOS>SUBSTRUCTURE")
    lines.append(f"     1 {resname:<4s}        1 RESIDUE           0 ****  ****    0 ROOT")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_tool(
    executable: str, arguments: list[str], *, work_dir: Path, output: Path
) -> AmberToolRun:
    found = which_executable(executable)
    if found is None:
        raise AmberToolsError(f"AmberTools executable not found: {executable}")
    result = run_command([found, *arguments], cwd=work_dir)
    stdout = work_dir / f"{executable}_stdout.txt"
    stderr = work_dir / f"{executable}_stderr.txt"
    stdout.write_text(result.stdout, encoding="utf-8")
    stderr.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0 or not output.is_file():
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-20:])
        raise AmberToolsError(
            f"{executable} failed with exit code {result.returncode}"
            f"{'' if output.is_file() else f' and wrote no {output.name}'}. "
            f"Command: {' '.join(result.command)}. See {stdout} and {stderr}.\n{tail}"
        )
    return AmberToolRun(
        command_result=result, stdout_path=stdout, stderr_path=stderr, output_path=output
    )


def _amberhome() -> Path:
    from mdprep.metals.nonbonded import NonbondedMetalError, resolve_amberhome

    try:
        return resolve_amberhome()
    except NonbondedMetalError as exc:
        raise ResidueParameterisationError(str(exc)) from exc
