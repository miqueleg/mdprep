# Examples

The top-level YAML files are manifest examples and are validated by the test
suite. Most reference placeholder input paths so users can copy and adapt them
to real systems.

| File | Purpose | Notes |
| --- | --- | --- |
| `01_protein_only_ff19sb.yaml` | Protein-only ff19SB setup | Schema-only example |
| `02_manual_catalytic_protonation.yaml` | Manual protonation overrides | Schema-only example |
| `03_multi_ligand_am1bcc.yaml` | Multiple AM1-BCC ligands | Requires AmberTools for real run |
| `04_qmmesp_pyscf_ligand.yaml` | PySCF QMMESP-style ligand charges | Requires AmberTools, tleap, ParmEd, PySCF |
| `05_histidine_gxtb_sp.yaml` | g-xTB histidine tautomer ranking | Requires xTB/g-xTB executable |
| `06_disulfide_manual.yaml` | Manual disulfide definition | Schema-only example |
| `07_gas_resp_pyscf_ligand.yaml` | PySCF gas RESP/ESP ligand charges | Requires AmberTools and PySCF |

The `tutorials/minimal_user_mol2/` directory contains a tiny toy example with
input PDB, ligand mol2, frcmod, and manifest. It is intended only for checking
installation and workflow mechanics, not for scientific interpretation.
