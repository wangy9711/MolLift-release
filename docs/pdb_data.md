# PDB-derived peptide training data

This document summarizes the construction protocol for the private PDB-derived
training dataset. The dataset and a package-format specification are not provided.

## Construction protocol used for the released model

The original package contained 211,247 fragments: 174,063 training, 22,209 validation
and 14,975 strict temporal-OOD test fragments. Fragments were continuous standard
peptide sequences of 5–50 residues, allowing MSE→MET, with heavy atoms only.
The source structures were X-ray/cryo-EM with resolution at most 3.0 Å.

Splitting was performed on complete chains before fragment extraction. Pre-2023
chains formed the train/validation pool, separated by sequence-cluster and PDB-entry
groups. Post-2023 test chains were excluded if sequence searches found a match to
the older pool at the joint thresholds of 30% identity and 80% bidirectional
coverage. Exact-sequence exclusion was applied as an additional safeguard. 

Extraction required continuous peptide connectivity and usable backbone/side-chain
coordinates, resolved alternate locations, and checked external covalent connections.
MSE selenium was mapped to MET sulfur. Interior fragments used heavy-atom ACE/NME
caps derived from flanking residues. Experimental atoms and synthetic caps have
separate masks. The sampled target mix aimed for 20% helix, 20% sheet, 30% coil and
30% mixed secondary structure, with integer fragment lengths balanced where feasible.

The original tools were Gemmi 0.7.5, mkdssp 4.2.2 and MMseqs2 18.8cc5c. 


