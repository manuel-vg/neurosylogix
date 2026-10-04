# Hybrid Models for Natural Language Reasoning: The Case of Syllogistic Logic
This repository contains the code and experimental resources for our study of logical generalization in large language models.
Using an extended syllogistic logic benchmark, we investigate two aspects of reasoning—compositionality and recursiveness—and show that, overall, LLMs perform reasonably well on recursive reasoning but struggle with compositional generalization. We further find that generalization is structure-dependent, with performance varying across different syllogistic structures. Finally, we propose a hybrid architecture that combines the efficiency of neural computation with the reliability of symbolic reasoning.

## Installation

- Download the GitHub repository.
- Install the environment using **environment.yml**.

## Dataset Generation

From the repository root, execute:

```bash
bash scripts/data_generation.sh T5|GPT pbc|ps ove|com|rec
```

where:

- `T5|GPT` specifies the model.
- `pbc|ps` specifies the task.
- `ove|com|rec` specifies the experiment.

### Example

```bash
bash scripts/data_generation.sh T5 ps ove
```

This generates the dataset for training and evaluating the T5 model on the premise selection (`ps`) task using the baseline (`ove`) experiment, where the model is trained and evaluated on all lengths.

## Training and Evaluation

The experiment script is available for the T5 model. 
GPT experiments are run using a separate API. See the [GPT-4o mini API documentation](https://developers.openai.com/api/docs/models/gpt-4o-mini) for instructions.

From the repository root, execute:

```bash
bash scripts/run_experiment.sh train|eval pbc|ps ove|com|rec 1|2|3
```

where:

- `train|eval` specifies the mode, either training or evaluation.
- `pbc|ps` specifies the task.
- `ove|com|rec` specifies the experiment.
- `1|2|3` specifies the run number.

### Example

```bash
bash scripts/run_experiment.sh train ps ove 1
```

This starts the first training run for the T5 model on the premise selection (`ps`) task using the baseline (`ove`) experiment.

## Citation

If you use this code or the resources in your research, please cite our paper:

```bibtex
@inproceedings{KR2026-107,
    title     = {{Hybrid Models for Natural Language Reasoning: The Case of Syllogistic Logic}},
    author    = {Vargas Guzmán, Manuel and Szymanik, Jakub and Malicki, Maciej},
    booktitle = {{Proceedings of the 23rd International Conference on Principles of Knowledge Representation and Reasoning}},
    pages     = {1143--1152},
    year      = {2026},
    month     = {7},
    doi       = {10.24963/kr.2026/107},
    url       = {https://doi.org/10.24963/kr.2026/107},
}
