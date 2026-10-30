"""Noncanonical amino acids kept inside a protein chain.

A modified residue of a chain -- a phosphoserine, an acetyl-lysine, a
p-azido-phenylalanine -- can be replaced by its standard parent
(``structure.repair.replace_modified_residues``) or kept as it is, under
``noncanonical_residues``. A kept residue is built from a template AmberTools
ships for it (``templates.py``) or parameterised from its SMILES
(``parameterise.py``), and bonded into the chain by tleap either way.
"""
