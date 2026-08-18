from pathlib import Path

from mdprep.config.models import ManifestConfig
from mdprep.refinement.selection import select_refinement_regions
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord
from tests.test_config_validation import base_manifest


def _atom(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resid: int,
    x: float,
    element: str,
    record_name: str,
) -> AtomRecord:
    return AtomRecord(
        serial=serial,
        name=name,
        altloc=None,
        resname=resname,
        chain_id=chain,
        resid=resid,
        icode=None,
        x=x,
        y=0.0,
        z=0.0,
        occupancy=1.0,
        bfactor=0.0,
        element=element,
        record_name=record_name,  # type: ignore[arg-type]
        original_line="",
    )


def _structure(path: str, atoms: list[AtomRecord]) -> PdbStructure:
    grouped: dict[tuple[str, str, int, str | None], list[AtomRecord]] = {}
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)
    residues = [
        ResidueRecord(
            id=ResidueId(chain_id=chain, resname=resname, resid=resid, icode=icode),
            atoms=residue_atoms,
            record_names={atom.record_name for atom in residue_atoms},
            original_index=index,
        )
        for index, ((chain, resname, resid, icode), residue_atoms) in enumerate(grouped.items())
    ]
    return PdbStructure(path=Path(path), atoms=atoms, residues=residues, model_count=1)


def _manifest() -> ManifestConfig:
    data = base_manifest()
    data["ligands"] = [
        {
            "id": "substrate",
            "selector": {"chain": "B", "resname": "SUB", "resid": 501},
            "net_charge": 0,
            "multiplicity": 1,
            "atom_types": "gaff2",
            "charge_method": "am1bcc",
        }
    ]
    data["metals"] = [
        {
            "id": "zinc",
            "model": "nonbonded",
            "ions": [
                {
                    "selector": {
                        "chain": "Z",
                        "resname": "ZN",
                        "resid": 1,
                        "atom_name": "ZN",
                    },
                    "element": "Zn",
                    "charge": 2,
                    "multiplicity": 1,
                }
            ],
            "nonbonded": {"parameter_set": "12_6"},
        }
    ]
    data["refinement"] = {
        "enabled": True,
        "qm_components": {
            "ligands": ["substrate"],
            "metal_sites": ["zinc"],
            "metal_coordinating_residues": {
                "zinc": [{"chain": "A", "resname": "ASP", "resid": 10}]
            },
        },
        "ash": {"python_executable": "/opt/ash/bin/python"},
    }
    return ManifestConfig.model_validate(data)


def test_qm_charge_uses_four_angstrom_residues_and_eight_angstrom_full_waters():
    leap_input = _structure(
        "input.pdb",
        [
            _atom(1, "N", "ASP", "A", 10, 1.0, "N", "ATOM"),
            _atom(2, "OD1", "ASP", "A", 10, 1.5, "O", "ATOM"),
            _atom(3, "CA", "ALA", "A", 11, 7.5, "C", "ATOM"),
            # tleap is allowed to move loaded ligands/ions ahead of waters.
            _atom(4, "O", "WAT", "W", 20, 7.5, "O", "HETATM"),
            _atom(5, "O", "WAT", "W", 21, 10.3, "O", "HETATM"),
            _atom(6, "C1", "SUB", "B", 501, 0.0, "C", "HETATM"),
            _atom(7, "O1", "SUB", "B", 501, 0.5, "O", "HETATM"),
            _atom(8, "ZN", "ZN", "Z", 1, 2.0, "Zn", "HETATM"),
        ],
    )
    topology = _structure(
        "provisional.pdb",
        [
            _atom(1, "N", "ASP", "", 1, 1.0, "N", "ATOM"),
            _atom(2, "OD1", "ASP", "", 1, 1.5, "O", "ATOM"),
            _atom(3, "CA", "ALA", "", 2, 7.5, "C", "ATOM"),
            _atom(4, "C1", "SUB", "", 3, 0.0, "C", "HETATM"),
            _atom(5, "O1", "SUB", "", 3, 0.5, "O", "HETATM"),
            _atom(6, "ZN", "ZN", "", 4, 2.0, "Zn", "HETATM"),
            _atom(7, "O", "WAT", "", 5, 7.5, "O", "HETATM"),
            _atom(8, "H1", "WAT", "", 5, 7.6, "H", "HETATM"),
            _atom(9, "H2", "WAT", "", 5, 7.4, "H", "HETATM"),
            _atom(10, "O", "WAT", "", 6, 10.3, "O", "HETATM"),
            _atom(11, "H1", "WAT", "", 6, 10.4, "H", "HETATM"),
            _atom(12, "H2", "WAT", "", 6, 10.2, "H", "HETATM"),
        ],
    )

    selection = select_refinement_regions(
        leap_input_structure=leap_input,
        topology_structure=topology,
        manifest=_manifest(),
    )

    assert selection.qm_residue_indices == (0, 4, 5)
    assert selection.total_qm_charge == 1  # substrate 0 + Zn(II) 2 + ASP -1
    assert selection.total_qm_multiplicity == 1
    assert selection.qm_boundary_excluded_atom_indices == (5,)
    assert selection.active_residue_indices == (0, 2, 4, 5)
    assert {6, 7, 8}.issubset(selection.active_atom_indices)
    assert 2 not in selection.active_atom_indices  # distant ALA
    assert {9, 10, 11}.isdisjoint(selection.active_atom_indices)  # water beyond 8 Å
    assert selection.active_region_cutoff_angstrom == 4.0
    assert selection.active_water_cutoff_angstrom == 8.0
    assert selection.movable_atom_policy == "all_active_region"


def test_hydrogen_only_policy_keeps_full_qm_region_but_moves_only_active_hydrogens():
    leap_input = _structure(
        "input.pdb",
        [
            _atom(1, "OD1", "ASP", "A", 10, 1.5, "O", "ATOM"),
            _atom(2, "HD1", "ASP", "A", 10, 1.6, "H", "ATOM"),
            _atom(3, "O", "WAT", "W", 20, 7.5, "O", "HETATM"),
            _atom(4, "C1", "SUB", "B", 501, 0.0, "C", "HETATM"),
            _atom(5, "H1", "SUB", "B", 501, 0.1, "H", "HETATM"),
            _atom(6, "ZN", "ZN", "Z", 1, 2.0, "Zn", "HETATM"),
        ],
    )
    topology = _structure(
        "provisional.pdb",
        [
            _atom(1, "OD1", "ASP", "", 1, 1.5, "O", "ATOM"),
            _atom(2, "HD1", "ASP", "", 1, 1.6, "H", "ATOM"),
            _atom(3, "C1", "SUB", "", 2, 0.0, "C", "HETATM"),
            _atom(4, "H1", "SUB", "", 2, 0.1, "H", "HETATM"),
            _atom(5, "ZN", "ZN", "", 3, 2.0, "Zn", "HETATM"),
            _atom(6, "O", "WAT", "", 4, 7.5, "O", "HETATM"),
            _atom(7, "H1", "WAT", "", 4, 7.6, "H", "HETATM"),
            _atom(8, "H2", "WAT", "", 4, 7.4, "H", "HETATM"),
        ],
    )
    manifest = _manifest().model_copy(
        update={
            "refinement": _manifest().refinement.model_copy(
                update={"movable_atoms": "active_region_hydrogens"}
            )
        }
    )

    selection = select_refinement_regions(
        leap_input_structure=leap_input,
        topology_structure=topology,
        manifest=manifest,
    )

    assert {0, 2, 4}.issubset(selection.qm_atom_indices)
    assert set(selection.active_atom_indices) == {1, 3, 6, 7}
    assert all(topology.atoms[index].element == "H" for index in selection.active_atom_indices)
    assert not set(selection.qm_atom_indices).issubset(selection.active_atom_indices)
    assert selection.movable_atom_policy == "active_region_hydrogens"
