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

`colab_smoke_test.ipynb` is a self-contained Google Colab installation test for
the PyPI `0.2.0` wheel. It tests the pip-installable Python package only;
complete Amber workflows also require the documented Conda-installed external
executables.

`mdprep_interactive_colab.ipynb` is the guided Colab system builder. Upload it
to Colab and run the cells in order. It can download an RCSB PDB entry or accept
an uploaded PDB, provides 3D review before and after explicit cleanup, proposes
reviewable RDKit ligand hydrogens, collects component charge/spin and mdprep
settings, validates the generated manifest, and downloads a portable input
bundle. PDB 7JSD (the BesD substrate complex) is loaded by default as a worked
structural example; its chemistry fields are intentionally not prefilled.
The implementation is included as one ordinary, documented Python cell—there
is no encoded payload or runtime module loader. Colab presents that run-once
cell collapsed so the scientific workflow remains readable, while an initial
miniature 3D view checks rendering before preparation starts.
It does not launch an application or widget dashboard: configuration uses
native Colab `#@param` forms and ordinary `input()` prompts, while every
structure is rendered directly in the output of the current Colab cell.
SMILES hydrogen templates must encode the requested formal charge explicitly
(for example, carboxylate `[O-]` rather than neutral acid `O`). The notebook
shows RDKit's interpreted charge before changing coordinates and allows a
failed template to be corrected without restarting the component workflow.
The current RCSB AKG CCD entry is neutral; the BesD dianion example therefore
uses the explicitly reviewed `O=C([O-])C(=O)CCC(=O)[O-]` SMILES.

The wizard offers explicitly selected nonbonded Amber ion models or one
single-chain bonded MCPB site. Bonded preparation prints distance-based
candidates for inspection but never turns them into bonds: the user must enter
and approve every atom selector, QM component, and QM/MM link-boundary
authorization. All multi-selections use commas. The capped MCPB model charge
and multiplicity and complete QM-region multiplicity are calculated from the
reviewed component charge/multiplicity data and explicit protein protonation
overrides, then printed for audit. A system with multiple open-shell components
and unspecified spin coupling fails clearly rather than receiving a guessed
multiplicity. Coordinating non-protein residues use
`mcpb_resp_pyscf` so MCPB's joint RESP charges replace their provisional
AM1-BCC charges. The Hessian can use pinned g-xTB or PySCF B3LYP/6-31G*; both
routes use an explicitly configured GFN2/ASH QM/MM-refined geometry.

The notebook's optional runtime cell installs AmberTools/MCPB.py, PropKa,
standard xTB, PySCF, OpenMM, geomeTRIC, and a pinned ASH revision with
micromamba/pip so a validated manifest can also be executed in Colab. A
separate cell installs the pinned official g-xTB Linux release, verifies its
archive and executable checksums, and must complete a real `--gxtb --hess`
smoke calculation before its executable path is offered for an MCPB manifest.
Standard conda-forge xTB is not treated as proof that g-xTB is available.
