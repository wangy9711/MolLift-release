# Model assets

The two pretrained checkpoints are included directly in the Git repository:

```text
checkpoints/
  mollift-drugs.pt
  mollift-peptide.pt
```


| Public name | Epoch | Role |
|---|---:|---|
| mollift-drugs | 197 | GEOM-Drugs Mixed, EMA |
| mollift-peptide | 20 | Retention-feasible PDB Mixed, EMA |

GEOM training uses explicit H. Peptide training uses heavy atoms, with 25% explicit-H
Drugs replay. The input JSON explicitly declares its atom representation; the
pipeline never silently adds or removes modeled atoms. 
