Inputs: bigquery_tables/edges.csv, bigquery_tables/nodes.csv
Outputs:
- graph.graphml (drug→disease), disease_disease_projection.graphml, drug_drug_projection.graphml
- summary_diseases_by_frequency.csv, summary_drugs_by_admissions.csv
- top_diseases_per_drug.csv, top_drugs_per_disease.csv
- disease_disease_projection.csv (shared_drugs ≥ 2, Jaccard), drug_drug_projection.csv (shared_diseases ≥ 2)
Notes: weights=admission counts; patients=unique subject counts; Jaccard on neighbor sets.
