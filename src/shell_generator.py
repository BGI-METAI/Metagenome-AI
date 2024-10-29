import os
from config import ConfigsGenerator


class ShellGenerator:

    def __init__(self, config, config_generator: ConfigsGenerator):
        """
        Initializes the SHGenerator with paths and templates for generating .sh scripts.

        Args:
            config (dict): Main configuration dictionary containing workdir path.
            config_generator (ConfigsGenerator): Instance of ConfigsGenerator for accessing configs and run info.
        """
        self.workdir = config.get("workdir")
        self.config_generator = config_generator
        self.sh_dir = os.path.join(self.config_generator.run_directory, "sh")
        self.sh_template = """#!/bin/bash
#DSUB -n {job_name}
#DSUB -N 1
#DSUB -A root.project.P24Z10200N0983
#DSUB -R "cpu=12;gpu=1;mem=32000"
#DSUB -oo train_classifier/%J.out
#DSUB -eo train_classifier/%J.err

# load module
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module add compilers/kgcc/10.3.1
module add libs/openblas/0.3.18_bs2.4.0
module add libs/cudnn/8.2.1_cuda11.3
module add compilers/cuda/11.3.0

ENV={env_path}
WORKDIR={workdir}
export PATH=$ENV:$PATH
cd $WORKDIR
python src/train.py -c {config_file}
"""

    def generate_sh_file(self, config):
        """
           Generates a single .sh file for the given configuration.

           Args:
               config (dict): Configuration dictionary containing `env`, `model_base_name`, `run_mode`, and `config_path`.
        """
        env_path = config["env"]
        model_base_name = config["model_base_name"]
        run_mode = config["run_mode"]
        config_file = config["config_path"]

        job_name = f"{run_mode}_{model_base_name}"
        sh_content = self.sh_template.format(
            job_name=job_name,
            env_path=env_path,
            workdir=self.workdir,
            config_file=config_file
        )
        sh_filename = f"{model_base_name}.sh"
        sh_file_path = os.path.join(self.sh_dir, os.path.join(run_mode, sh_filename))
        os.makedirs(os.path.dirname(sh_file_path), exist_ok=True)

        # Write SSH content to file
        with open(sh_file_path, 'w') as file:
            file.write(sh_content)

    def generate(self):
        """
        Iterates through all configs and generates .sh files for each based on run mode and model specifics.
        """

        # Create run mode directories for organizational structure if not existing
        for run_mode in self.config_generator.RUN_MODES:
            os.makedirs(os.path.join(self.sh_dir, run_mode), exist_ok=True)

        configs = self.config_generator.get_configs()

        # Generate an .sh file for each configuration in the list
        for config in configs:
            self.generate_sh_file(config)


if __name__ == "__main__":
    config = {
        "models": [
            {
                "model_type": "ESM",
                "max_tokens": 2000,
                "batch_size": 64,
                "env": "conda/esm"
            }
        ],
        "datasets": [
            {
                "dataset_name": "toxisity",
                "train": "/home/share/huadjyin/home/chenjunhong/META_AI/datasets/tox_merged/tox_train.tsv",
                "test": "/home/share/huadjyin/home/chenjunhong/META_AI/datasets/tox_merged/tox_test.tsv",
                "valid": "/home/share/huadjyin/home/chenjunhong/META_AI/datasets/tox_merged/tox_validation.tsv"
            },
            {
                "dataset_name": "G+",
                "train": "/home/share/huadjyin/home/chenjunhong/META_AI/datasets/tox_merged/tox_train.tsv",
                "test": "/home/share/huadjyin/home/chenjunhong/META_AI/datasets/tox_merged/tox_test.tsv",
                "valid": "/home/share/huadjyin/home/chenjunhong/META_AI/datasets/tox_merged/tox_validation.tsv"
            }
        ],
        "classifiers": [
            {
                "classifier_type": "MLP",
                "batch_size": 64,
                "num_epochs": 60,
                "lr": 1e-3,
                "hidden_layers": None,
                "early_stop_patience": 4,
            }
        ],
        "run_name": "test1",
        "work_dir": "caslav/test"

    }

    print("Creating Generator")
    confGen = ConfigsGenerator(config)
    print("Started generating")
    confGen.generate()
    print("Finished generating")

    print("Generating sh files")
    shGen = ShellGenerator(config, confGen)
    shGen.generate()
    print("Files Generated")
