# BayesAge2.0

## BayesAge v2.0 (10/14/2024)

BayesAge 2.0 is a framework for epigenetic (epiage) and transcriptomic age (tAge) predictions. This is an extension to our previous BayesAge model that only focused on epiage predictions from Mboning et al published in Frontiers in Bioinformatics. 
BayesAge utilizes maximum likelihood estimation (MLE) to infer ages, models count data using binomial distributions for bulk bisulfite data and poisson distribution for gene expression data, and uses LOWESS smoothing to capture the non-linear dynamics between methylation/gene expression level and age.<br>
For more information on the algorithm, please consult Mboning et al., "BayesAge: A Maximum likelihood estimation algorithm to predict epigenetic age", "BayesAge 2.0: A Maximum Likelihood Algorithm to Predict Transcriptomic Age". <br>

If you use this software, please cite our works along with Trapp's work (scAge).

The BayesAge pipeline consists of two steps for tAge predictions: <br>
    1) Computing the nonlinear models for each gene within a gene expression matrix using LOWESS fit. <br>
    2) Predicting age of samples given number of observed counts per gene from a single sample in a tsv file. <br>

and can be executed with the following functions: <br>
    1) transcriptome_reference <br>
    2) pdAge <br>
    
The BayesAge pipeline consists of three steps for epiage predictions: <br>
    1) Computing the nonlinear models for each CpG within a DNAm matrix using LOWESS fit. <br>
    2) Loading in and processing sample methylation cgmap files.  <br>
    3) Predicting age of samples given number of cytosines and coverage from cgmap files.

and can be executed with the following functions: <br>
    1) epigenome_reference <br>
    2) process_cgmap_file <br>
    3) bdAge <br>  

## Installation & Usage <br>

To install BayesAge 2.0 and associated data, please clone the GitHub repository:

git clone https://github.com/lajoycemboning/BayesAge2.0.git

This will download all required data to utilize and test the software.

For ease of use, all functions needed to run the full BayesAge pipeline are directly included within the BayesAge2.py script. <br>

To run BayesAge, add the directory containing BayesAge2.py to your path, then import BayesAge2 into a Python script or Jupyter notebook as shown in the example notebook.

In order to use the functions provided in <b>BayesAge2</b>, the following packages need to be installed:

'numpy' (tested with version 1.24.3) <br>
'pandas' (tested with version 2.1.1) <br>
'scipy' (tested with version 1.11.2) <br>
'tqdm' (tested with version 4.66.1) <br>
'statmodels' (tested with version 0.14.0) <br>

This tool was developed in Python 3.10.12 in a Jupyter Notebook environment, running on Ubuntu 22.04.3 LTS.

To visualize epiage or tAge predictions, installation of 'seaborn', 'matplotlib' and 'statannot' is highly recommended.

Predicted transcriptomic age are conveniently written to .csv files and epigenetic age output dataframes are conveniently written .tsv files, and can therefore also be analyzed in any other environment (R, Excel, etc...).

## Speed, Memory Use and Parallelization <br>
The five main functions (*epigenome_reference*, *process_cgmap_file*, *bdAge*, *transcriptome_reference*, *pdAge*) are fully functional running on a single core (the default is n_cores = 1). Since this package is a direct extension of scAge, they should experience linear speedup with multiprocessing. Increasing the number of cores (n_cores) will improve the speed and efficiency of the pipeline. While developing this package, I used a System76 laptop.

process_cgmap_file and bdAge could eat up a lot of RAM depending on how many samples are being processed simulatenously, so one can monitor the memory allocation when running the algorithm.

## Example Notebooks <br>
The notebook *How_to_Run_BayesAge_to_predict_tAge.ipynb* contains instructions on how to run the BayesAge pipeline for transcriptomic age predictions.<br>
Please refer to this main notebook on how to train the model (*transcriptome_reference*) and predict tAge (*pdAge*). <br>

Below is a brief overview of the parameters that each functions require to run properly.

### Training using transcriptome_reference <br>

In order to train the reference matrix for the BayesAge pipeline, you can run the function:

```
transcriptome_reference(training_matrix="/home/lajoyce/Documents/BayesAge2.0/brain/loocv_sample/brain_sample_1.csv",
reference_name="brain_reference_1.csv",
output_path="/home/lajoyce/Documents/BayesAge2.0/brain/reference/",
age_prediction="list",
age_list=[1, 3, 6, 9, 12, 15, 18, 21, 24, 27],
min_age=1,
max_age=24,
age_step=1,
tau=0.7)
```
where: 
* `training_matrix` --> full path to the reference matrix you want to use to create the trained reference matrix. <br>
* `reference_name` --> name of the processed reference matrix dataset. <br>
* `output_path` --> the full path where to output the reference matrix files. <br>
* `age_prediction` --> choice between "age_steps" vs. "list" to make the reference age bins. <br>
* `age_list` --> if *age_prediction* is set to *list*, use this parameter to give a specific list to create the age bins of the reference model. <br>
* `min_age` --> if *age_prediction* is set to *age_steps*, set minimum age of the reference model. <br>
* `max_age` --> if *age_prediction* is set to *age_steps*, set maximum age of the reference model. <br>
* `age_step` --> age step between minimum age and maximum age. <br>
* `tau` --> Tau value of the lowess fit. <br>

### Predicting age using pdAge <br>

We can predict the age of samples using the 'pdAge' function:

```
 pdAge(prediction_matrix="/home/lajoyce/Documents/BayesAge2.0/brain/test/sample1.csv",
    sample_name="sample1",
reference_data="/home/lajoyce/Documents/BayesAge2.0/brain/reference/brain_reference_sample",
    output_path="/home/lajoyce/Documents/BayesAge2.0/brain/predictions/",
    selection_mode="numGenes",
    gene_parameter=12,
    age_prediction="list",
    age_list = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27],
    min_age=0,
    max_age=100,
    age_step=1,
    n_cores=1,
    chunksize=5)
```
where:
* `prediction_matrix` --> directory containing matrix for sample phi we want to predict the age of.
* `sample_name` --> the desired name of the sample file with predicted age.
* `reference_data` --> The full file path to the desired reference data/model made from step 1.
* `output_path` --> path of directory to store the files with predicted ages.
* `selection_mode` --> One of the selection modes to select the top genes sites (numGenes, percentile, cutoff).
* `gene_parameter` --> Parameter to specifically choose the number of genes.
* `age_prediction` --> choice between "age_steps" vs. "list" to make the age steps for prediction.
* `age_list` --> if *age_prediction* is set to *list*, use this parameter to give a specific list to create the age steps for age prediction(should be same as reference model). <br>
* `min_age` --> if *age_prediction* is set to *age_steps*, set the minimum age for which to build a probability profile.
* `max_age` --> if *age_prediction* is set to *age_steps*, set the maximum age for which to build a probability profile.
* `age_step` --> The step value for computing probability profiles.
* `n_cores` --> The number of cores to use for parallel processing.
* `chunk_size` --> The number of elements to feed to each worker during parallel processing.

## Data <br>
The raw counts and metadata (*MACA_modified_rawcounts.csv.gz*, *MACA_modified_metadata.csv*) used in this study are included in the folder for example purposes and for reproducibility.

## Troubleshooting <br>

If you encounter any issues when trying to run BayesAge, or if you have suggestions, please feel free to contact me by email: lajoycemboningatucla.edu.

