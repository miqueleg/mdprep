"""A configured metal ion inside an xTB histidine cluster.

The cluster around a metal-site histidine contains the metal. Its charge is
declared in the manifest under `metals`, but the cluster accounting only knew
about `ligands`, so preparing any metalloenzyme whose histidines sit near the
metal stopped with "Unknown heterogen ... configure it under ligands". Found by
running BesD (6NIE), where the iron is 2.2 A from two histidines.
"""

import pytest

from mdprep.config.models import ManifestConfig
from mdprep.protonation.histidine_xtb import (
    HistidineXtbError,
    cluster_charge_breakdown,
    estimate_cluster_charge,
)
from mdprep.structure.pdb import read_pdb


STRUCTURE = "tests/data/protein_histidine_zinc.pdb"


def manifest_with(metals: list[dict], ligands: list[dict] | None = None) -> ManifestConfig:
    return ManifestConfig.model_validate(
        {
            "project": {
                "name": "t",
                "input_structure": STRUCTURE,
                "output_dir": "prepared/t",
            },
            "structure": {},
            "protein": {"forcefield": "ff14SB", "water_model": "TIP3P"},
            "protonation": {"method": "manual_only"},
            "disulfides": {},
            "ligands": ligands or [],
            "metals": metals,
            "solvation": {},
            "validation": {},
        }
    )


ZINC_SITE = {
    "id": "zn",
    "model": "nonbonded",
    "ions": [
        {
            "selector": {
                "chain": "Z",
                "resname": "ZN",
                "resid": 500,
                "icode": None,
                "atom_name": "ZN",
            },
            "element": "Zn",
            "charge": 2,
        }
    ],
    "nonbonded": {"parameter_set": "12_6"},
}


def cluster() -> list:
    return list(read_pdb(STRUCTURE).residues)


def test_a_configured_metal_ion_contributes_its_declared_charge():
    terms = cluster_charge_breakdown(cluster(), manifest_with([ZINC_SITE]), {})

    metal_terms = [t for t in terms if t["source"] == "configured_metal_ion"]
    assert len(metal_terms) == 1
    assert metal_terms[0]["charge"] == 2
    assert "ZN" in str(metal_terms[0]["residue"])


def test_the_declared_oxidation_state_is_used_not_a_guess():
    """A charge of +1 must be honoured even though zinc is usually +2."""
    site = {**ZINC_SITE, "ions": [{**ZINC_SITE["ions"][0], "charge": 1}]}

    terms = cluster_charge_breakdown(cluster(), manifest_with([site]), {})

    metal = next(t for t in terms if t["source"] == "configured_metal_ion")
    assert metal["charge"] == 1


def test_the_metal_charge_reaches_the_cluster_total():
    with_metal = estimate_cluster_charge(cluster(), manifest_with([ZINC_SITE]), {})
    neutral_site = {**ZINC_SITE, "ions": [{**ZINC_SITE["ions"][0], "charge": 1}]}
    with_one = estimate_cluster_charge(cluster(), manifest_with([neutral_site]), {})

    assert with_metal - with_one == 1


def test_an_unconfigured_metal_still_fails_clearly():
    """Silence here would mean an xTB cluster with the wrong electron count."""
    with pytest.raises(HistidineXtbError, match="Unknown heterogen"):
        cluster_charge_breakdown(cluster(), manifest_with([]), {})


def test_the_error_mentions_both_places_a_heterogen_can_be_configured():
    with pytest.raises(HistidineXtbError, match="metals"):
        cluster_charge_breakdown(cluster(), manifest_with([]), {})


# --------------------------------------------------------------------------
# Spin state
# --------------------------------------------------------------------------


def test_a_closed_shell_cluster_has_no_unpaired_electrons():
    from mdprep.protonation.histidine_xtb import cluster_unpaired_electrons

    closed = {**ZINC_SITE, "ions": [{**ZINC_SITE["ions"][0], "multiplicity": 1}]}

    assert cluster_unpaired_electrons(cluster(), manifest_with([closed])) == 0


def test_high_spin_iron_two_contributes_four_unpaired_electrons():
    """xTB will not converge a closed-shell solution for high-spin Fe(II)."""
    from mdprep.protonation.histidine_xtb import cluster_unpaired_electrons

    iron = {
        **ZINC_SITE,
        "ions": [{**ZINC_SITE["ions"][0], "element": "Zn", "multiplicity": 5}],
    }

    assert cluster_unpaired_electrons(cluster(), manifest_with([iron])) == 4


def test_a_metal_without_a_declared_multiplicity_fails_clearly():
    """A spin state cannot be inferred from an element."""
    from mdprep.protonation.histidine_xtb import cluster_unpaired_electrons

    with pytest.raises(HistidineXtbError, match="multiplicity"):
        cluster_unpaired_electrons(cluster(), manifest_with([ZINC_SITE]))


def test_the_spin_reaches_the_xtb_command_line():
    from mdprep.config.models import HistidineXtbConfig
    from mdprep.protonation.xtb_runner import build_xtb_command

    command = build_xtb_command(
        config=HistidineXtbConfig(),
        xyz_path="cluster.xyz",
        cluster_charge=2,
        unpaired_electrons=4,
        executable="xtb",
    )

    assert "--uhf" in command
    assert command[command.index("--uhf") + 1] == "4"


def test_a_cluster_with_no_metal_stays_closed_shell():
    from mdprep.config.models import HistidineXtbConfig
    from mdprep.protonation.xtb_runner import build_xtb_command

    command = build_xtb_command(
        config=HistidineXtbConfig(),
        xyz_path="cluster.xyz",
        cluster_charge=0,
        executable="xtb",
    )

    assert command[command.index("--uhf") + 1] == "0"
