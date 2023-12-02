# Functional Prediction of Somatic Missense Variants Using Large Protein Language Model

This is the project repository for the project "Functional Prediction of Somatic Missense Variants Using Large Protein Language Model". Here are some brief description of the key items in the repository

tl;dr: the `data/unknown_data_merged_12_01_23.tsv` file contains the GoLFNet predicted score for somatiuc missense mutations with unknown OncoKB annotations.

## Jupyter notebooks:

### `esm_model.ipynb`
The main notebook which performs the analysis outlined in the report. Note that the default data path may need to be updated for the code to execute properly

### `data_preprocessing.ipynb`
The data preprocessing steps that were done to convert raw data to model-ready data. Note that this is not extensively tested since the data has already been generated. The processed data are all avilalble under the `data` directory.

### `visualization_R.ipynb`
The jupyter notebook that was used to generate some of the figures in the report. Note that it uses a R kernel instead of a python kernel.

## The `/data` directory:
The directory that stores the data needed for model training as well as results. 

### Model evaluation related:
`train_data_fasta_slim.tsv` and `test_data_fasta_slim.tsv`: The processed training and testing data for the models tested

`hugo_to_uniprot.tsv`: The mapping used for mapping gene names (hugo symbols) to the corresponding uniprot id provided in the original AlphaMissense paper.

`uniprot_sequences.tsv`: The amino acid sequences retreived from the Uniprot API when the uniprot ID is queried.

`unknown_to_test_recurrent_only.tsv`: Somatic missense mutations in the MSK MetTropism cohort that appeared in at least 2 samples. 

### (Model evaluation related data that are not provided)

`AlphaMissense_aa_substitutions.tsv`: The file that stores all the alpha missense score from the AlphaMissense paper. Can be downloaded directly from the data availability section of the original paper: https://www.science.org/doi/10.1126/science.adg7492

`msk_met_2021/`: The directory that contains all the results from the MSK MetTropism study. Can be downloaded from CBioPortal at: https://www.cbioportal.org/study/summary?id=msk_met_2021

### Results Related

`model_performances.tsv` and `model_performances_long.tsv`: The performance of the evaluated models in different data structure. Used for visualization downstream.

`unknown_data_merged_12_01_23.tsv`: The GoLFNet predicted scores for somatic missense mutations with no current OncoKB annotations. 

### Visualization Related

`gene_aa_pivot.tsv`: The data needed to visualize the landscape fo mutation annotation

`label_known_aa_covered.tsv`: The data needed to visualize the training/testing set up.

## The `/scripts` directory:
The directory that contains all the helper functions / classes used in the project. Each method is commented and the file names should eb self-explanatory so a detailed description is not provided here.
