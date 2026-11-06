# Attribution and data resources

EviDrug is the current implementation in this repository. The package retains the equations and selected parameter tensors of the frozen EviDrug study while separating the encoder, clinical-memory backbone, evidence refinement and training controller into direct modules.

The twelve published-method comparison results identify their source papers, years, venues and checkpoint hashes in `results/baseline_registry.csv`. This focused release implements EviDrug and its component/weight controls; it does not distribute the historical baseline training workspace.

MetaDrug is a comparison method. Its original implementation is available at <https://github.com/aryahm1375/MetaDrug>. This repository does not relicense or redistribute files from that upstream repository. Literature references and implementation differences are recorded in `references.bib` and the result documentation.

Public NDC/ATC mappings and the GAMENet interaction table are obtained separately from the resource URLs in `configs/resource_checksums.json`. Dataset access, mappings and third-party resources retain their original terms. Neither clinical records nor licensed dataset archives are included in this software release.

