# LLMs-for-SPARQL
This repository contains the code and datasets for our study, ***Are LLMs adequate SPARQL query generators?
 Investigating Zero-Shot NL-to-SPARQL translation***. The project evaluates the capability of Large Language Models
 (LLMs) to generate SPARQL queries from natural language (NL) questions in a zero-shot setting.



## Overview
We **test multiple LLMs on diverse knowledge graphs with different ontologies: given an NL question and ontology data,
 the LLM models generate SPARQL queries, which are then executed and compared against ground truth queries to assess
 their accuracy**. Our evaluation explores:
- The impact of ontology context on query generation
- The performance of different prompt templates
- How model size affects query correctness and complexity handling

This repository provides the **full experimental pipeline**, including:
- Datasets (NL questions, SPARQL queries, ontologies)
- LLM inference scripts for query generation
- Evaluation metrics for result comparison



## Datasets
We use two datasets for evaluating NL-to-SPARQL translation:
- **[Spider4SPARQL](https://github.com/ckosten/Spider4SPARQL/)**  
- **[Bestiary](https://github.com/danrd/sparqlgen/tree/main)**  

Unlike many NL-to-SPARQL datasets based on DBpedia or Wikidata, these datasets are built on **custom, less popular
 knowledge graphs**, making them ideal for testing in a **zero-shot** setting.  

### Why these datasets?  
1. **Reduced model bias** – since LLMs are less likely to have encountered these KGs during training, they have fewer
 opportunities to "cheat."  
2. **Realistic custom KG use case** – they better simulate scenarios where users generate queries for a freshly
 designed KG with no training data.  
3. **Compact ontologies** – their small schemas allow us to include the **entire ontology** in the prompt without
 needing retrieval or filtering, avoiding extra pre-processing steps that could impact evaluation.  

### Dataset structure  
The datasets are stored in the **`datasets/`** directory:  
- **`datasets/bestiary/`**  
- **`datasets/spider4sparql/`**  
  - **`processed/`** – Contains the enhanced versions of the datasets.  
  - **`raw/`** – Contains the original versions of the datasets (to be downloaded separately).  
  - **`scripts/`** – Includes code and README files detailing the modifications.  

### Raw datasets  
The original **unmodified datasets were too large for GitHub**, so we provide them on **[Zenodo](https://zenodo.org/records/14978788)**.  
The archive includes the same structure as the **`datasets/`** directory, with the **`raw/`** folders containing the
 original data. Modifications to the datasets were tracked locally using **Git**, and the `.git` file is included in
 the archive.

### Dataset modifications  
We made several modifications to improve the ontology quality of the datasets. Here are some examples:  
- **Spider4SPARQL** – Added **domain axioms** and renamed **ambiguous classes/properties** for clarity.  
- **Bestiary** – Added **new predicates (e.g., `beast_category`)** to replace information previously extracted from
 URIs (which was non-RDF-compliant).  

For a full list of modifications and details on how they were made, please refer to the README files in the
 **`scripts/`** directories of each dataset folder:
  - [Spider4SPARQL - preprocessing](datasets/spider4sparql/scripts/preprocessing/README.md)
  - [Spider4SPARQL - ontology refinement](datasets/spider4sparql/scripts/onto_refinement/README.md)
  - [Bestiary - enhancement](datasets/bestiary/scripts/README.md)


## SGPT baseline  
We used **[SGPT](https://github.com/rashad101/SGPT-SPARQL-query-generation/tree/main)** as a baseline for comparison in
our experiments. **SGPT is a tool for SPARQL query generation from natural language**, originally trained and tested on
datasets such as *LC-QUAD2*, *QUALEX9*, and *VQUANDA*. Since ***Bestiary* doesn't have a training set**, we only
ran *SGPT* on *Spider4SPARQL*.

To adapt *SGPT* for use with *Spider4SPARQL*, we modified its code to ensure compatibility with the training and development
sets of this dataset. The tool is run the same way as before, but with the additional parameter `spider4sparql` to
specify the new dataset.

The modified *SGPT* code is not included in this repository. Instead, you can download our modified version and the
 model trained on *Spider4SPARQL* from **[Zenodo](https://zenodo.org/records/14978788)**. After downloading, place the
 *SGPT* code in a `sgpt/` folder in this repository.



## Source folder  
The **`src/`** folder contains several utility modules and frameworks used in our experiments:
- **`llms/`** – custom framework we built to facilitate easy invocation of multiple LLMs for testing and running
- **`logger`** – generic logging utility for tracking and debugging.  
- **`jena.py`** – Python interface to invoke *Jena* and efficiently run SPARQL queries on RDF Knowledge Graphs.
- - **`timeout.py`** – functions to run code with a maximum execution time.
 experiments. This package helps streamline the process of experimenting with various LLMs and comparing their results.  



## Experiments
The **`experiments/`** folder contains the code to replicate our experiments and the results. It is structured as follows:  

- **`agents/`** – code to run the LLMs on the query generation task. The modules **`basic.py`** and **`detailed.py`**
 contain the two prompt templates we used in our experiments.  
- **`evaluation/`** – contains the evaluation scores, metrics, and comparison criteria for query results.  
- **`logs/`** and **`runs/`** – contain the output from the LLMs' query generation and other processes.

Additionally, there are four key scripts:  
- **`run_ground_truth.py`** – runs all the ground truth queries and stores their output.  
- **`run_sgpt_processing.py`** – takes the output generated by running *SGPT* on *Spider4SPARQL* and organizes it into the
 same data structure we used for the other LLMs (facilitating processing).  
- **`run_llm_generation.py`** – runs an LLM (specified with macros in the code, no CLI parameters yet) over both
 datasets and both prompt styles.  
- **`run_evaluation.py`** – runs the evaluation process, comparing the LLM-generated queries against the ground truth
 and computing various scores such as accuracy, generation time, syntax correctness, and determinism (as detailed in
 the paper).

### Results
The following is a preview of the accuracies obtained by the various tested LLMs:  
![ENSEMBLE accuracies](experiments/evaluation/scores/plot_ensemble_accuracy.png)

More detailed explanations and comments about the accuracy results and other metrics can be found in the paper.
The full evaluation results are available in the **`experiments/evaluation/scores/`** directory.



### How to Run
This project requires **Python 3.12**. We provide a **`requirements.txt`** file with the necessary dependencies. You
 can install them using:
```bash
pip install -r requirements.txt
```
We used `transformers==4.48.1` for most of the project, but downgraded to `transformers==4.47.0` to run
 `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`.

Once the dependencies are installed, ensure that the source folder `src` is included in your `PYTHONPATH` environmental
 variable. Then, each script can be run from its respective folder using the following command:
```bash
python <script.py>
```
For example, to run the ground truth queries:
```bash
python run_ground_truth.py
```

### Requirements to run SPARQL queries
To run queries using *Jena* through the `jena.py` interface, you will need
**[Apache Jena](https://jena.apache.org/download/index.cgi)** installed on your system.
Additionally, make sure to set the `JENA_HOME` environment variable to the location where Jena is installed.

### Notes about datasets
**Important:** to run the scripts in the **`datasets/`** folder, you will need to download the raw datasets. The
 original versions of the datasets are available on **[Zenodo](https://zenodo.org/records/14978788)**. Ensure that the
 raw data is in the correct directory before running the associated scripts.

Make sure to check each script's README or documentation for any additional details or setup specific to the script.
