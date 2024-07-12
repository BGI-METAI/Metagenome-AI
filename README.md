# Metagenome-AI

## Overview

Provide a brief introduction to the project and its purpose.

## Configuration Parameters

1. `sequences_path` (String): Path to the file containing the protein sequences to be processed.

2. `emb_dir` (String): Directory where the embeddings of the protein sequences will be stored. If you only train the model, provide here the path to the directory with saved embeddings.

3. `program_mode` (Integer): Specifies the mode in which the program will run. Possible modes: ONLY_STORE_EMBEDDINGS (just store embeddings), TRAIN_PREDICT_FROM_STORED (just train classifier from stored embeddings), or RUN_ALL (store embeddings and train classifier). Default is `RUN_ALL`. 

4. `train` (String): Path to the training dataset in CSV format.

5. `test` (String): Path to the test dataset in CSV format.

6. `valid` (String): Path to the validation dataset in CSV format.

7. `emb_type` (String): Type of embeddings to be generated for the protein sequences. Availale: ESM, PTRANS, 

8. `prot_trans_model_path` (String): Path to the pre-trained protein transformer model. If null, the model will be loaded from the specified name.

9. `prot_trans_model_name` (String): Name of the pre-trained protein transformer model to be used. Default is `prot_t5_xl_uniref50`.

10. `max_tokens` (Integer): Maximum number of tokens to be processed per batch. Default is `3000`.

11. `batch_size` (Integer): Number of sequences to be processed in each batch. Default is `40`.

12. `num_epochs` (Integer): Number of epochs for training the model. Default is `20`.

13. `lr` (Float): Learning rate for the training process. Default is `1e-3`.

14. `label` (String): Column name in the dataset that contains the labels. Default is `label`.

15. `sequence` (String): Column name in the dataset that contains the sequences. Default is `original`.

16. `max_seq_len` (Integer or Null): Maximum sequence length. If null, no maximum length is enforced. Default is `null`.

17. `model_folder` (String): Directory where the trained model weights will be saved. Default is `weights`.

18. `model_basename` (String): Base name for the saved model files. Default is `prot_model_`.

19. `preload` (String): Path to the pre-trained model weights to be loaded. If null, training starts from scratch. Default is `null`.

20. `experiment_name` (String): Name of the experiment, used for tracking and logging purposes. Default is`runs/esmSmall`

21. `wandb_key` (String): API key for Weights & Biases (wandb) used for logging and tracking experiments.

## Installation

Provide instructions on how to install and set up the project, including any dependencies.

## Usage

Explain how to use the project, including examples of how to configure the parameters in the configuration file.

## Contributing

If you accept contributions, provide guidelines for how others can contribute to the project.

## License

Specify the project's license.

## Contact

Provide contact information or links to relevant resources (e.g., issue tracker, mailing list).
