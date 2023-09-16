# Master thesis: Using nD-Laplace to train privacy-preserving clustering algorithms on distributed n-dimensional data​

Experiments are all collected in the main_new.py file inside the ExperimentRunners folder.
**Mechanisms:**
- nD-Laplace
- Piecewise
- grid-nD-Laplace
- density-nD-Laplace

**Datasets:**
- heart-dataset
- seeds-dataset
- circle-dataset
- skewed-dataset
- line-dataset
  
**Separated into different runners:**
- Generate input data: Run to generate all data for all privacy budgets for four mechanisms. Outputs the perturbed data for all mechanisms:
`generate-input-data {dataset}`
- Generate utility data: Collects all data for external/internal validation.
`generate-utility-experiments {dataset}`
- Generate distance data: Collects all data for privacy distance.
`generate-distance-data {dataset}`
- Generate privacy data: Collects all data for the membership inference attacks.
`generate-security-experiments {dataset}`

**Export data for thesis:**
Command:
`generate-thesis-reports {dataset}`
