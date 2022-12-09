import os
from typing import List, Dict
import time
from matplotlib import pyplot as plt


class ResultPrinter:

    def __init__(self, param_label: str, runs: Dict[str, Dict[str, float]], time_label: str):
        """
        Initializes the result printer. Creates result output directory structure.
        :param param_label: should uniquely identify each hyper parameter setting.
        :param runs: dictionary to store the dictionaries of validation metrics for the final epoch of each run. Should
        be shared by all ResultPrinter instances for a model tuning session.
        """
        self.param_label = param_label
        self.runs = runs
        self.parent_path = 'C:\\Users\\shawn\\Desktop\\Development\\CS7643' # add - must use abs path
        # Create the directory if it does not exist
        self.base_path = os.path.join(self.parent_path, 'auto_results', time_label)  # add
        self.run_path = os.path.join(self.base_path, param_label)  # add
        print('run path', self.run_path)
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)
        self.out_file = open(f"{self.run_path}\\results.txt", "w")  # add

    def print(self, print_str: str, end='\n') -> None:
        """
        Prints a line to the output file for this parameter setting.
        :param print_str: the string to print.
        :param end: a end char
        :return: None
        """
        print(print_str, end=end)
        self.out_file.write(print_str + end)

    def rankAndSave(self, validation_metrics: Dict[str, float]) -> None:
        """
        Prints a file at the base path of the run with all run validation metrics per param setting ranked
        lowest ot highest by validation loss.
        :param validation_metrics: the dictionary of validation metrics for the final epoch of this run
        :return: None
        """
        self.runs[self.param_label] = validation_metrics
        # sort_by lowest validation loss
        with open(f"{self.base_path}ranked_results.txt", "w") as rank_file:
            for out in sorted(self.runs.items(), key=lambda x: x[1]['f1_score'], reverse=True):  # add key
                rank_file.write(str(out) + '\n')

    def makePlots(self, training_losses: List[float], validation_losses: List[float], epoch: int):
        """
        Saves the training and validation plots per epoch and prints their exact values to the output file for
        this parameter setting.
        :param training_losses:
        :param validation_losses:
        :param epoch:
        :return:
        """
        plt.clf()
        plt.plot(range(len(training_losses)),training_losses,'r')
        plt.plot(range(len(validation_losses)),validation_losses,'b')
        plt.legend(['Training Loss','Validation Loss'])
        plt.title('Training and Validation Loss for Unet')
        plt.savefig(f"{self.run_path}plot{epoch}.txt")
        # write loss arrays to outfile
        self.print(f"training loss per epoch: {str(training_losses)}")
        self.print(f"validation loss per epoch: {str(validation_losses)}")

    def close(self):
        self.out_file.close()



