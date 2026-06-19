# Project: language model distillation

## Goal

Create a codebase for scientific understanding of language model distillation, specifically the pareto frontier of size vs performance.

## File structure
- `train.py`: the file for a training run.
  - This supports two modes:
    1. training directly from the data, in an autogressive fashion over all tokens.
    2. distillation, i.e. training using a teacher model.
      - the loss objective is a sum of both the loss w.r.t. data and w.r.t. the teacher model, i.e. `alpha * CE_loss + (1-alpha) * KL_divergence`.
      - the training data is offline, and the teacher logits are pre-computed and loaded from the disk.
  - For each mode, the loss can be configured to be either over all tokens, or only over answer tokens.
  - If `cfg.data.eval_files` is a list of files, then create one loader for each file and evaluate on each loader separately.

- `evaluate.py`: the file for an inference run. This contains
  - `if __name__ == '__main__'` which allows to run the file directly.
  - The `def eval` function which can be imported in `train.py` to monitor performance during training. 
    - This supports an option to save all the generations and/or the logits, which will be used in distillation training.
  - The accuracy should be only over answer tokens. The loss can be either over answer-only tokens or over all tokens, depending on the config.

- `./models/`: a folder containing model-related python files.
  - `./models/get_model.py`: This supports different models, including:
    - the Qwen3 family, specifically 8B, 4B, 1.7B, 0.6B, base and instruct.
    - the SmolLM2 family, specifically 1.7B, 360M, 135M, base and instruct.

- `./data/`: a folder containing data-related python files. This includes a range of synthetic tasks, where each task corresponds to one file.
  - `./data/__init__.py`: this includes:
    - a master class `class SyntheticTask` for synthetic tasks.
      - each subclass must inherit the function `def generate_samples`, which is called at instantiation together with the number of samples in the dataset `n_samples` and returns a list of samples.
      - each subclass has a function `save_samples` which can optionally be called after a list of samples is obtained by `generate_samples`.
    - a `tokenize` function that tokenizes the saved text samples. A tokenizer needs to be specified, which can be the Qwen3 tokenizer or the SmolLM2 tokenizer.
    - a `get_loader` function which takes in a list of file paths; each file path is a tokenized file.
  - `./data/memorization.py`: this is a memorization task on (A, B) relations.
    - At initialization, create `n_pairs` pairs of (A, B) relations. To construct A or B, first sample a random adjective (e.g. 'medical') and then sample a random noun (e.g. 'puppy'). The phrases don't need to make semantic sense. To ensure there are no duplicates in A or B with high probability, each phase is appended with a 8-character random string, i.e. each A or B is of the format "{adj}-{noun}-{rnd string}".
      - Importantly, train and validation/test dataset share the same set of phrases; in other words, this task tests for in-weight memorization.
    - To generate a sample:
      - First sample `T` uniformly from `min_T` and `max_T`, inclusive.
      - Sample `T-1` relations from the pool of `n_pairs` relation pairs. For each relation, write it either as "{A} owns {B}" or "{B} is owned by {A}", with equal probability.
      - Sample another (A,B) relation, and a query that is either "What does {A} own?" or "Who owns {B}?" with equal probability. The answer is "{B}" or "{A}" respectively.
        - Note that some of these queries are not answerable early in training.
        - But these relations will have been seen multiple times during training with high probability, so the model will gradually be able to answer the relation query.
  - `./data/line_graph.py`: this is a reasoning task.
    - This class has a fixed attribute which is a set of 500 unique names, containing female, male, and gender-neutral names.
    - At initialization, do the following:
      - Set the min and max sequence length `min_T` and `max_T`.
    - To generate samples: 
      - Sample a sequence length `T` uniformly from `[min_T, max_T]` inclusive.
      - Sample a length-`T` line graph. Specifically, sample `T` unique names from the 500 names, and label them as 0, 1, ..., T-1.
      - Turn this line graph into a sentence. Specifically, randomly shuffle the `T-1` edges, and translate each edge as `{name_a} is taller than {name_b}` if `a` is the left-end of the edge, or `{name_a} is shorter than {name_b}` if `a` is the right-end of the edge. Then, sample two names `x,y` and ask "Is {x} talled than {y}?" The answer is "Yes" if `x` is to the left of `y` on the line graph, and "No" otherwise.
  - `./data/s5.py`: this is a reasoning task for the symmetric group of order 5, which can be considered as tracking the permutation states of 5 objects.
    - This class has a fixed attribute which is a set of 500 nouns. These nouns can be any objects or animals. They don't have to make semantic sense.
    - At initialization, do the following:
      - Sample a set of 100 random nouns from the size-500 set, from which objects will be drawn when defining samples.
      - Get a list of generators for S5, which includes the following actions:
        - Swap the first two elements;
        - Shift all element by 1 to the left;
        - and `n_actions - 2` number of distinct random swaps, where the two indices being swapped are randomly sampled as 5-choose-2, except if the two indices are 0,1 or 1,0 in which case we re-sample the swap.
      - Assign probabilities to each action following what's given by the config.
      - Set the min and max sequenth length `min_T` and `max_T`.
    - Note that the probabilities and set of objects can be different in train and test sets.
    - To generate a sample:
      - First, sample `T` actions according to the action distribution, where `T` is sampled uniformly from `[min_T, max_T]` inclusive. Set the initial permutation to [0, 1, 2, 3, 4], and compute the final permutation after seeing all the actions.
      - Then, translate the actions, the initial permutation, and the final permutation into natural languages. The samples are of the format "There are 5 slots, containing {obj_1}, {obj2}, {obj3}, {obj4}, {obj5} from slot 1 to 5, respectively. Someone {action_1}. Someone then {action_2} ... Finally, someone {action_T}. What objects do the 5 slots contain?",
        - `obj_1` to `obj_5` are sampled from the size-100 set, each corresponding to the initial indices 0, 1, 2, 3, 4.
        - `action_t` is of the following two forms:
        - If it's a shift, then say "shift all the objects to the left by 1 slot."
        - If it's a swap, then say "swap objects in slot {x} and slot {y}" where x, y are given by the swap.

- `./configs/`: this contains yaml files for configuration.
  - Include all hyperparameters for optimizers, data, and models.

- `./scripts/`: a folder containing bash scripts for running expeirments. For example, `scripts/train.sh` runs `python train.py`.




- `./analysis/`: a folder containing files for examining results and error analysis.
  - `check_generations.ipynb`: a Jupyter notebook, that allows to take in a file saved by `save_generations=True` in `evaluate.py`, and the generations.

## Requirements
- Use wandb for logging.
- Use Hydra for hyperparameter config.
- Before running the code, activate `conda activate common`.
- For generating data, make it so that:
  - There is a config yaml file specifying the hyperparameters for all tasks.
  - Add a script to `scripts/generate_by_yaml.sh` which takes in the yaml file, and save the generated data files in a way such that:
    - The directory is `data_files/{yaml_file_name_without_extension}/`. The yaml file is included in the directory.
    - The directory contains the following files:
      - The jsonl files `{task}_{split}.jsonl`, for every task and `split` being "train" or "eval".
      - (TODO) The tokenized files:
        - `{model_family}_train.pt`, where `model_family` is "qwen3" or "smollm2"; this file merges over all tasks.
        - `{model_family}_{task}_eval.pt`, where `model_family` is "qwen3" or "smollm2"; there is one file per task, unless the task has 0 samples.
