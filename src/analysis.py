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
        # amp_df.head()
        amp_subset_df = amp_df

        # Define custom color palette to have shades of the same color per model
        color_palette = {
            'PTRANS_MLP_3HL': 'yellow', 'PTRANS_XGBoost': 'orange',
            'EMS2_650M_MLP_3HL': 'lightgreen', 'EMS2_650M_XGBoost': 'green',
            'EMS2_3B_MLP_3HL': 'lightcoral', 'EMS2_3B_XGBoost': 'red',
            'ESM3_MLP_3HL': 'lightblue', 'ESM3_XGBoost': 'blue'
        }

        # Sort data based on the preferred model order
        model_order = ['PTRANS_MLP_3HL', 'PTRANS_XGBoost', 'EMS2_650M_MLP_3HL', 'EMS2_650M_XGBoost', 'EMS2_3B_MLP_3HL',
                       'EMS2_3B_XGBoost', 'ESM3_MLP_3HL', 'ESM3_XGBoost']
        amp_subset_df['Base model'] = pd.Categorical(
            amp_subset_df['Base model'], categories=model_order, ordered=True
        )
        amp_subset_df = amp_subset_df.sort_values('Base model')

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 7))
        barplot = sns.barplot(data=amp_subset_df[amp_subset_df['Metric'] == 'MCC'], x='Dataset', y='Score',
                              hue='Base model', palette=color_palette)
        plt.ylim([50, 100])
        plt.grid(zorder=0, linewidth=1, linestyle=':', color='k')
        # Configure the legend to be on a single line and in the specified location
        plt.legend(
            prop={'size': 11}, ncol=len(color_palette.keys()), loc='upper center',
            bbox_to_anchor=(0.5, -0.15), fancybox=True
        )
        plt.box(False)

        # Annotate each bar with the value inside
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.1f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, 9), textcoords='offset points',
                             fontsize=10, fontweight='bold')  # Set font size and bold
        save_fig_path = os.path.join(self.analysis_dir, f'testing_models3.png')
        plt.savefig(save_fig_path, dpi=200, bbox_inches='tight')

    def testing_models_one_metric(self):
        # amp_df = pd.read_excel('../data/predictions/ripp_lab_validation/AMP_test.xlsx', sheet_name='AMP')
        amp_df = pd.read_csv(self.metrics_table_path)
        # amp_df.head()
        amp_subset_df = amp_df[amp_df['Dataset'] == 'AMP']
        amp_subset_df = amp_subset_df[amp_subset_df['Metric'] == 'MCC'].sort_values(by='Score', ascending=True)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 7))
        barplot = sns.barplot(data=amp_subset_df[amp_subset_df['Metric'] == 'MCC'], x='Dataset', y='Score',
                              hue='Base model')
        plt.ylim([50, 100])
        plt.grid(zorder=0, linewidth=1, linestyle=':', color='k')
        plt.legend(
            prop={'size': 11}, ncol=len(amp_subset_df.keys()), loc='upper center',
            bbox_to_anchor=(0.5, -0.15), fancybox=True
        )
        plt.box(False)

        # Annotate each bar with the value inside
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.1f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, 9), textcoords='offset points',
                             fontsize=10, fontweight='bold')  # Set font size and bold
        save_fig_path = os.path.join(self.analysis_dir, f'testing_models.png')
        plt.savefig(save_fig_path, dpi=200, bbox_inches='tight')

    def analyse(self):
        os.makedirs(self.analysis_dir, exist_ok=True)
        # Step0: Create Table from all models metrics
        self.create_metric_table()
        # Step1: Testing Models
        self.testing_models()
        # self.testing_models_one_metric()


