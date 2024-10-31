import glob
import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib_venn import venn3
import seaborn as sns
import numpy as np
from venny4py.venny4py import *
from config import ConfigsGenerator


class Analyser:
    def __init__(self, config_generator: ConfigsGenerator):
        self.config_generator = config_generator
        self.analysis_dir = config_generator.get_directory_path("analysis")
        self.metrics_table_path = "metrics_table.csv"

    def create_metric_table(self):
        classifiers_path = self.config_generator.get_directory_path("classifiers")
        results_table = pd.DataFrame(columns=["Base model", "Score", "Dataset", "Metric"])

        # Find all .csv files in classifiers_path and its subdirectories
        csv_files = glob.glob(os.path.join(classifiers_path, "**", "*.csv"), recursive=True)

        # Append each CSV's content to results_table
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            results_table = pd.concat([results_table, data], ignore_index=True)

        # Save the aggregated results_table to a CSV file
        save_path = os.path.join(self.analysis_dir, "metrics_table.csv")
        self.metrics_table_path = save_path
        results_table.to_csv(save_path, index=False)

    def testing_models(self):
        # amp_df = pd.read_excel('../data/predictions/ripp_lab_validation/AMP_test.xlsx', sheet_name='AMP')
        amp_df = pd.read_csv(self.metrics_table_path)
        #amp_df.head()
        amp_subset_df = amp_df

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(13, 7))
        barplot = sns.barplot(data=amp_subset_df[amp_subset_df['Metric'] == 'MCC'], x='Dataset', y='Score',
                              hue='Base model')
        plt.ylim([50, 100])
        plt.grid(zorder=0, linewidth=1, linestyle=':', color='k')
        plt.legend(prop={'size': 11})
        plt.box(False)

        # Annotate each bar with the value inside
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.1f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, 9), textcoords='offset points',
                             fontsize=12, fontweight='bold')  # Set font size and bold
        save_fig_path = os.path.join(self.analysis_dir, f'testing_models.png')
        plt.savefig(save_fig_path, dpi=200, bbox_inches='tight')

    def analyse(self):
        os.makedirs(self.analysis_dir, exist_ok=True)
        # Step0: Create Table from all models metrics
        self.create_metric_table()
        # Step1: Testing Models
        self.testing_models()


