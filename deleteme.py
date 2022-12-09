from utils.resultPrinter import ResultPrinter
from typing import Dict
import time

# runs dict should be passed to each instance of a results printer. It is only appended to so should be thread safe.
descrip_name = 'batch_size=14'
runs: Dict[str, Dict[str, float]] = {}
# create a new results printer for each param setting tested
time_label = int(time.time())
for i in range(5):
    descrip_name = 'batch_size=' + str(i)
    result_printer = ResultPrinter(descrip_name, runs, time_label)
    train_metrics = {'loss': i+1, 'f1': 1-i}
    result_printer.print(f'Training metrics: {str(train_metrics)}')
    valid_metrics = {'loss': i+10, 'f1': i-10}
    result_printer.rankAndSave(valid_metrics)