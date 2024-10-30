import os
import subprocess
import argparse
import warnings
import glob
from shell_generator import *


def submit_jobs(job_dir):
    """
    Submits all jobs in the specified directory with dsub -s and waits for them to finish.

    Args:
        job_dir (str): Directory containing .sh files to submit.
    """
    print(f"Submitting all jobs in {job_dir}...")
    processes = []
    for sh_file in glob.glob(os.path.join(job_dir, "*.sh")):
        process = subprocess.Popen(["dsub", "-s", sh_file])  # Assumes `dsub -s` command for job submission
        processes.append(process)

    # Wait for all jobs in the directory to finish
    for process in processes:
        process.wait()
    print(f"All jobs in {job_dir} finished.")


def run_analysis():
    """Runs the analysis.py script."""
    print("Running analysis.py...")
    subprocess.run(["python", "analysis.py"], check=True)
    print("Analysis completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all Analysis"
    )

    parser.add_argument("-c", "--config_path", type=str)

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    config = ConfigProviderFactory.get_config_provider(args.config_path)

    # Step 1: Generate Config

    confGen = ConfigsGenerator(config)
    print("Started generating config files...")
    confGen.generate()
    print("...Finished generating config files!")

    print("Started generating shell files...")
    shGen = ShellGenerator(config, confGen)
    shGen.generate()
    print("...Finished generating shell files!")

    base_sh_dir = shGen.sh_dir

    # Step 2: Submit and wait for jobs in the get_embeddings directory
    get_embeddings_dir = os.path.join(base_sh_dir, "create_embeddings")
    submit_jobs(get_embeddings_dir)

    # Step 3: Submit and wait for jobs in the train_classifiers directory
    train_classifiers_dir = os.path.join(base_sh_dir, "train_classifiers")
    submit_jobs(train_classifiers_dir)

    # Step 4: Run the analysis
    run_analysis()
