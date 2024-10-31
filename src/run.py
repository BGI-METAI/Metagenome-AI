import subprocess
import argparse
import warnings
from shell_generator import ShellGenerator
from config import ConfigsGenerator, ConfigProviderFactory
from analysis import Analyser
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
    job_ids = []
    for sh_file in job_dir:
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
    print("STEP1: Generating Config and Shell files")
    confGen = ConfigsGenerator(config)
    confGen.generate()

    shGen = ShellGenerator(config, confGen)
    shGen.generate()

    base_sh_dir = shGen.sh_dir

    run_mode = config.get("run_mode")
    if run_mode is None or len(run_mode) == 0 or run_mode == "RUN_ALL":
        run_mode = ["CREATE_EMBEDDINGS", "TRAIN_CLASSIFIERS", "ANALYSE"]

    # Step 2: Submit and wait for jobs in the get_embeddings directory
    if "CREATE_EMBEDDINGS" in run_mode:
        print("STEP2: CREATING EMBEDDINGS")
        submit_jobs(shGen.generated_sh_files["create_embeddings"])

    # Step 3: Submit and wait for jobs in the train_classifiers directory
    if "TRAIN_CLASSIFIERS" in run_mode:
        print("STEP3: TRAINING CLASSIFIERS")
        submit_jobs(shGen.generated_sh_files["train_classifiers"])

    # Step 4: Run the analysis
    if "ANALYSE" in run_mode:
        print("STEP4: ANALYSIS")
        analyser = Analyser(confGen)
        analyser.analyse()
