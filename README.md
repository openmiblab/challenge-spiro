![example-result](_static/spiro-logo.png)

---

# SPIRO challenge

## Fast mechanistic model inversion

[![Code License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square&logo=apache&color=blue)](https://www.apache.org/licenses/LICENSE-2.0) [![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) 

---

### :pushpin: Summary of the challenge

This repository provides a forward model function in python with the following signature: 

```python
signal = forward(parameters, time, **settings)
```

The `forward` model simulates an experiment defined by some `settings`. It aims to derive a number of  `parameters` from `signal` arrays measured at given `time` points. The challenge is to **train a deep-learning model that solves the inverse problem**. The model inference should be wrapped up in a function with the following signature:

```python
parameters = my_solution(time, signal, **settings)
```

Your solution will be ranked based on a *global score* that consists of two equally weighted parts: 
- *accuracy score*: compares `parameters` generated with your solution against unseen ground truth simulated with `forward`
- *generalizability score*: compares `signals` reconstructed with your solution against measured data in an unseen dataset.

---

## 🚀 Installation

To install the required materials, clone or download this repository and navigate to the top folder `challenge-spiro`. 

For installation with conda (recommended):

```
conda env create -n spiro -f env.yml
conda activate spiro
```

For installation with pip (alternative), create and activate a virtual environment and run:

```
pip install -e .
```

You can test your installation by running the case study notebook *docs/case_study.ipynb*. This is a also a good way to familiarise yourself with some of the functionality included in this distribution, and it showcases some diagnostics that will be useful when you evaluate your solution.

---

## 🛠️ Testing your solution

The challenge distribution includes two solutions that you can use as benchmarks, as well as some functionality to create a dummy score for your solution and the benchmarks. The dummy score is exactly the same as the official score, except that it uses known, public data rather than unseen data.

To test your solution, drop a single python module in the *solutions* folder. The module must contain a function with the required signature. To register your solution, include it in the `SOLUTIONS` dictionary in the `__init__.py` file of `challenge_spiro`. All solutions listed in this dictionary will be scored and added to the league table, so if you want to compare multiple solutions during development, just add them in there. 

Once your solution is registered in `__init__.py`, you can compute a dummy score by running:

```
python -m challenge_spiro.compute_dummy_scores
```

If you include the benchmark method in the `SOLUTIONS` register, this can take a few hours to compute. When it finishes, you will find a new folder *dummy_secrets* with the outputs, including a league table in csv format.

**Note (1)**: The challenge distribution also contains the script *compute_official_scores.py* which is used to compute the actual score for all submissions. The script is included for transparency but can only be run by organisers who have access to the `secrets` folder. 

**Note (2)**: The two existing solutions included with the challenge distribution are:
- `benchmark.py`: This is the least-squares iterative optimization method that is used in the publications. It is therefore a useful benchmark for evaluating your solution.
- `normative.py`: This is a solution which is used to normalize the score. It is of little value in practice as it returns constants for the key parameters. By definition, it should score 100%.

---
  
## 📤 Submission of solutions

To submit your solution, please create a single zip file containing:

1. Your python module with the function.
2. Trained model weights.
3. A requirements.txt file listing any requirements on top of those already included.
4. The league table with dummy scores so we can verify our local implementation.

Please send the zip file as an email attachment or a downloadable link to the challenge organisers.

**Note**: You do *not* need to submit the training script or the training data. 

---

## :file_folder: Code structure

The **challenge-spiro** distribution contains 3 top-level folders:
- **src**: The source code for the challenge with tools to generate synthetic data and compute scores, or visualise results. Apart from the forward model it also includes two examples of inverse solutions (not using deep learning).
- **docs**: A folder with a jupyter notebook illustrating some of the functionality included.
- **secrets**: This folder is only accessible to challenge organisers. It contains unseen experimental data as well as the unseen seed to generate the digital reference object that is used in the official score.

The **src** folder has the top level scripts to compute official scores (organisers only) and dummy scores. The subfolder **src.solutions** contains python modules with solutions and model weights needed to run them. The solutions can be imported as:

```python
from challenge_spiro.solutions.benchmark import inverse
```

The **src.utils** has key functionality in three modules:

- *model*: Defines the forward model and the plot function, which can be imported into python code with:

```python
from challenge_spiro.utils.model import forward, plot
```

- *dro*: uses the forward model and random number generators to generate a virtual subject (`generate_subject`) and a digital reference object including a virtual population with data and ground truths (`generate_dro`). There are used in the score and can also be used to generate new data:

```python
from challenge_spiro.utils.dro import generate_subject, generate_dro
```

- *score*: All functionality needed to compute the scores and save the results into a league table. The scoring is done by the top level function `score_all_submissions`:

```python
from challenge_spiro.utils.score import score_all_submissions
```
---

## 📚 Context

The data and modelling approach are adapted from the methods used in the following paper:

- Min et al. 2025. Assessing Drug-mediated Inhibition of Liver Transporter Function with MRI: A First-in-Human Study. Radiology 317(3) [[Link](https://pubs.rsna.org/doi/10.1148/radiol.251899)]

An open access preprint with more detail is here:

- Min et al. 2025. Assessing Drug-mediated Inhibition of Liver Transporter Function with MRI: A First-in-Human Study. medrxiv [[Link](https://www.medrxiv.org/content/10.1101/2025.06.16.25329670v1)]

The software implementation of the model is taken from the [dmri.org](https://dcmri.org/) python package, which also has examples showing these data and methods.


---

## 💰 Funder

The SPIRO project is funded by:

[![example-result](_static/ycr-logo.png)](https://www.yorkshirecancerresearch.org.uk/)