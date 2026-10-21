# Halogenating and Dehalogenating Enzymes

Enzymes that put a halogen on a carbon, or take one off, come from a dozen
families with little chemistry in common: non-heme iron, heme-thiolate and
heme-histidine centres, vanadate, flavins, SAM, corrinoids and iron-sulfur
clusters, and several with no cofactor at all. This page lists what mdprep
does with a representative of each, so that you know before you start whether
your enzyme is something it prepares, what it will ask, and where it stops.

Every representative below was loaded into the planner (`mdprep plan`, the web
builder and the Colab notebook all use it); the ones marked **prepared** were
also taken through the whole toolchain to a validated Amber topology.

| Family | Representative | What mdprep does |
| --- | --- | --- |
| Fe(II)/alpha-ketoglutarate halogenases | SyrB2 (2FCT), BesD (6NIE) | **Prepared.** Bonded Fe(II) site with the chloride and alpha-ketoglutarate in the MCPB model; examples 16-18. |
| Cytochrome P450 | CYP119 (1IO7) | **Prepared.** The heme's iron is read out into a residue of its own and bonded to the porphyrin, the cysteine thiolate and the water; example 19. |
| Unspecific peroxygenases (UPO) | AaeUPO (5OXU, 2YOR) | **Prepared.** Heme-thiolate as for a P450; the N-glycans and crystallisation additives removed on two questions; the magnesium on the heme's propionate nonbonded; example 20. |
| Chloroperoxidase | *C. fumago* CPO (1CPO) | Heme-thiolate; 21 sugars (N- and O-linked) removed; the N-terminal pyroglutamate replaced by glutamine; the manganese must be modelled nonbonded, since one bonded site is fitted per preparation. |
| Dehaloperoxidase | DHP A (2QFK) | A globin: heme bonded to the proximal histidine and a water; ammonium and sulfate additives removed. |
| Myeloperoxidase, lactoperoxidase | MPO (1CXP) | **Refused**: the heme is bonded to the protein through three side chains (a covalently attached cofactor). |
| Vanadium haloperoxidases | *C. inaequalis* VCPO (1IDQ) | **Refused**: vanadate is an oxoanion, not a metal site mdprep can build. |
| Flavin-dependent halogenases | PrnA (2AR8), RebH (2OA1) | **Prepared** (PrnA). FAD's SMILES and charge (-2) are proposed; the substrate or product is typed in; example 21. RebH's free tryptophan carries the amino acid's name and is offered a rename first. |
| SAM-dependent fluorinase and chlorinase | FlA (1RQR, 2V7V) | SAM, SAH proposed; 5'-fluoro-5'-deoxyadenosine typed in. 1RQR's selenomethionines are replaced by methionine on one question, and its free methionine is offered a rename. |
| Haloalkane dehalogenases | DhaA (4E46), DhlA (2DHC) | No cofactor: the additives are removed, the substrate is a ligand. |
| L-2-haloacid dehalogenases | 1ZRN | The ester intermediate on the catalytic aspartate is a heterogen bonded to the protein: removing it is asked, with nothing proposed. |
| Fluoroacetate dehalogenase | FAcD (3R3U) | The nickels from the His-tag purification can be removed on their model question. |
| Iodotyrosine deiodinase | IYD (4TTB) | FMN's SMILES and charge (-2) are proposed. |
| Reductive dehalogenases | RdhA (4UR0, 4RAS) | **Refused**: two [4Fe-4S] clusters, which no single metal site describes (as are the [2Fe-2S] clusters of Rieske oxygenases). |

## What a student meets on the way

- **Glycans.** Fungal peroxygenases are glycoproteins. Sugars bonded to the
  protein are found from the file's LINK records, or by bond length if it has
  none, and removing them is one question, proposed yes. See
  [heterogens.md](heterogens.md#glycans).
- **Crystallisation additives.** Glycerol, methanol, sulfate, phosphate, PEG:
  one question, proposed yes, every residue listed. Answer no to be asked
  about each as a ligand.
- **Modified residues.** Selenomethionine is replaced by methionine on one
  question. Anything else modified is asked with nothing proposed.
- **A metal that is not part of the system.** A His-tag nickel, a buffer
  cadmium: the metal's model question offers to remove it.
- **Magnesium and calcium** are proposed nonbonded.
- **Organic cofactors.** FAD, FMN, SAM, SAH, NAD+, NADH, NADP+ and NADPH are
  recognised by name and atoms, and their SMILES and charge at pH 7 are
  proposed with the protonation they assume. A reduced flavin (FADH-) is a
  different molecule: change the SMILES and the charge if that is the state you
  simulate.
- **Two metals and one heme.** In a peroxygenase the heme holds its iron and,
  through a propionate, a magnesium. The iron is fitted with MCPB.py and the
  heme's charges come from that fit; the magnesium stays a nonbonded ion, and
  the QM/MM refinement of the iron's site treats it as part of the MM
  environment.

## Where mdprep stops

Covalently attached cofactors (heme c, the hemes of myeloperoxidase and
lactoperoxidase), vanadate and other oxoanions, iron-sulfur clusters,
corrinoids together with their clusters, modified residues with no standard
parent, and protein-protein crosslinks other than disulfides are refused when
the structure is loaded, with the reason. Two metal sites that would each need
their own MCPB fit are refused at the question that makes them so.
