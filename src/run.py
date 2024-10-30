import os
import subprocess
import argparse
import warnings
import glob
from shell_generator import *
import time


def get_dict(output: str) -> dict:
    # Split the output into lines
    lines = output.strip().split('\n')

    # Extract headers from the first line and values from the second line
    headers = lines[0].split()
    values = lines[1].split()

    # Map headers to values to create the dictionary
    result_dict = {header: value for header, value in zip(headers, values)}

    return result_dict


def submit_jobs(job_dir):
    """
    Submits all jobs in the specified directory with dsub -s and waits for them to finish.

    Args:
        job_dir (str): Directory containing .sh files to submit.
    """
    print(f"Submitting all jobs in {job_dir}...")
    job_ids = []
    for sh_file in glob.glob(os.path.join(job_dir, "*.sh")):
        submit_command = f"dsub -s {sh_file}"
        result = subprocess.run(submit_command, shell=True, capture_output=True, text=True)
        # Capture job ID from submission response
        results_dict = get_dict(result.stdout)
        job_id = results_dict["JOBID"] # Modify this based on your dsub output
        job_ids.append(job_id)

    for job_id in job_ids:
        djob_command = f"djob {job_id}"
        while True:
            result = subprocess.run(djob_command, shell=True, capture_output=True, text=True)
            results_dict = get_dict(result.stdout)
            status = results_dict["JOB_STATE"]
            print(f"Job: {job_id} {status}")
            if status == "SUCCEEDED":
                break
            elif status == "FAILED":
                break
            else:
                # Waiting for jobs to complete...
                time.sleep(60)


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
    #submit_jobs(get_embeddings_dir)

    # Step 3: Submit and wait for jobs in the train_classifiers directory
    train_classifiers_dir = os.path.join(base_sh_dir, "train_classifiers")
    submit_jobs(train_classifiers_dir)

    # Step 4: Run the analysis
    #run_analysis()
