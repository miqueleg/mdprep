# Examples

The top-level YAML files are manifest examples and are validated by the test
suite. Most reference placeholder input paths so users can copy and adapt them
to real systems.

| File | Purpose | Notes |
| --- | --- | --- |
| `00_basic_protein.yaml` | Minimal ff14SB/TIP3P protein preparation | Same basic example shown in the README |
| `01_protein_only_ff19sb.yaml` | Protein-only ff19SB setup | Schema-only example |
| `02_manual_catalytic_protonation.yaml` | Manual protonation overrides | Schema-only example |
| `03_multi_ligand_am1bcc.yaml` | Multiple AM1-BCC ligands | Requires AmberTools for real run |
| `04_qmmesp_pyscf_ligand.yaml` | PySCF electrostatically embedded QMMESP charges | Requires AmberTools, tleap, ParmEd, PySCF |
| `05_histidine_gxtb_sp.yaml` | g-xTB histidine tautomer ranking | Requires xTB/g-xTB executable |
| `06_disulfide_manual.yaml` | Manual disulfide definition | Schema-only example |
| `07_gas_resp_pyscf_ligand.yaml` | PySCF gas RESP/ESP ligand charges | Requires AmberTools and PySCF |
| `08_nonbonded_zinc_1264.yaml` | Explicit Zn2+ Amber 12-6-4 model | Requires AmberTools and ParmEd |
| `09_bonded_zinc_mcpb_prepare.yaml` | Staged bonded Zn MCPB.py input generation | Requires AmberTools/MCPB.py; QM completion is a second run |
| `10_qmmm_refinement_ash.yaml` | ASH/GFN2-xTB active-site refinement before final QMMESP charges | Requires AmberTools, ASH, OpenMM, geomeTRIC, xTB, and PySCF for the final charge stage |
| `11_qmmm_metal_gfn2_mcpb.yaml` | Integrated metal-site QM/MM refinement, GFN2-xTB Seminario, MCPB RESP, and final MD build | Requires AmberTools/MCPB.py, ASH, OpenMM, geomeTRIC, xTB, and PySCF |
| `12_qmmm_metal_mace_polar1_mcpb.yaml` | Integrated metal-site refinement, MACE-POLAR-1 Hessian, MCPB RESP, and final MD build | Requires the example 11 tools plus a licensed MACE-POLAR-1 environment |
| `13_qmmm_refinement_gxtb_mechanical.yaml` | Method-specific g-xTB/OpenMM mechanical-embedding refinement | Requires AmberTools, ASH, OpenMM, geomeTRIC, and a g-xTB-enabled xTB binary |
| `14_qmmm_refinement_mace_polar1_mechanical.yaml` | Method-specific MACE-POLAR-1/OpenMM mechanical-embedding refinement | Requires AmberTools and an ASH environment containing OpenMM, geomeTRIC, and licensed MACE-POLAR-1 dependencies |
| `15_roe_brooks_openmm.yaml` | Roe--Brooks equilibration and a manifest-sized OpenMM production run | Uses a GPU when available and falls back to CPU |

The `tutorials/minimal_user_mol2/` directory contains a tiny toy example with
input PDB, ligand mol2, frcmod, and manifest. It is intended only for checking
installation and workflow mechanics, not for scientific interpretation.
