# Manifest Reference

Top-level keys:

- `project`
- `structure`
- `protein`
- `protonation`
- `disulfides`
- `ligands`
- `solvation`
- `validation`

## project

- `name`: project identifier.
- `input_structure`: PDB input path.
- `output_dir`: mdprep output directory.

## structure

- `keep_crystal_waters`: keep waters in the input PDB.
- `altloc_policy`: `highest_occupancy`, `first`, or `fail`.
- `remove_unknown_heterogens`: remove unconfigured heterogens instead of
  failing.
- `preserve_chain_ids`: preserve chain IDs when possible.
- `remove_input_hydrogens`: remove input hydrogens before protonation-stage PDB
  output.

## protein

- `forcefield`: `ff14SB` or `ff19SB`.
- `water_model`: `TIP3P` or `OPC`.

## protonation

- `ph`: target pH.
- `method`: `manual_only`, `propka`, or `propka_xtb_his`.
- `overrides`: manual residue-state assignments.
- `histidine.xtb`: xTB/g-xTB settings for HID/HIE ranking.

## disulfides

- `auto_detect`: detect close CYS/CYX SG-SG pairs.
- `detection_cutoff_angstrom`: SG-SG cutoff.
- `force`: manually forced disulfide pairs.
- `forbid`: pairs that must not be auto-assigned.

## ligands

Each ligand has:

- `id`
- `selector`
- `net_charge`
- `multiplicity`
- `atom_types`: `gaff` or `gaff2`
- `charge_method`: `am1bcc`, `user_mol2`, `gas_resp_pyscf`, or
  `qmmesp_pyscf`
- optional `user_mol2`
- optional `user_frcmod`
- preservation controls for names and coordinates
- optional `qmmesp` block for PySCF charge workflows

## qmmesp

- `qm_engine`: must be `pyscf`.
- `method`: `HF` or a PySCF DFT functional string.
- `basis`: PySCF basis.
- `embedding_cutoff_angstrom`: MM point-charge cutoff for QMMESP.
- `grid`: deterministic Connolly-like ESP grid settings.
- `resp_fitting`: native RESP/ESP-like fitting settings.
- `environment`: include/exclude protein, waters, and other ligands.

The target ligand is always excluded from its own MM embedding.

## solvation

- `enabled`
- `box`: `truncated_octahedron` or `rectangular`
- `buffer_angstrom`
- `neutralize`
- `salt_concentration_molar`
- `positive_ion`
- `negative_ion`

## validation

- `run_openmm_energy_check`
- `fail_on_warnings`
- `fail_on_missing_parameters`
- `fail_on_noninteger_ligand_charge`
