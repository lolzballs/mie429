import dataclasses
import queue
import threading
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy


CHRONOLOGICAL_AGE = {
    'M': [3/12, 6/12, 9/12, 12/12, 18/12, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    'F': [3/12, 6/12, 9/12, 12/12, 18/12, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
}
BONE_AGE = {
    'M': [3.01, 6.09, 9.56, 12.74, 19.36, 25.97, 32.40, 38.21, 43.89, 49.04, 56.00, 62.43, 75.46, 88.20, 101.38, 113.90, 125.68, 137.32, 148.82, 158.39, 170.02, 182.72, 195.32, 206.21],
    'F': [3.02, 6.04, 9.05, 12.04, 18.22, 24.16, 30.96, 36.63, 43.50, 50.14, 60.06, 66.21, 78.50, 89.30, 100.66, 113.86, 125.66, 137.86, 149.62, 162.28, 174.25, 183.62, 189.44]
}
STD_DEV = {
    'M': [0.69, 1.13, 1.43, 1.97, 3.52, 3.92, 4.52, 5.08, 5.40, 6.66, 8.36, 8.79, 9.17, 8.91, 9.10, 9.00, 9.79, 10.09, 10.38, 10.44, 10.72, 11.32, 12.86, 13.05],
    'F': [0.72, 1.16, 1.36, 1.77, 3.49, 4.64, 5.37, 5.97, 7.48, 8.98, 10.73, 11.65, 10.23, 9.64, 10.23, 10.74, 11.73, 11.94, 10.24, 10.67, 11.30, 9.23, 7.31]
}

BOUND_LOW = {}
BOUND_HIGH = {}

for sex, vals in BONE_AGE.items():
    in_years = [month/12 for month in vals]
    BONE_AGE[sex] = in_years

for sex, vals in STD_DEV.items():
    sd_years = [month/12 for month in vals]
    BOUND_LOW[sex] = [age - 2*sd if (age - 2*sd) > 0 else 0 for age, sd in zip(CHRONOLOGICAL_AGE[sex], sd_years)]
    BOUND_HIGH[sex] = [age + 2*sd for age, sd in zip(CHRONOLOGICAL_AGE[sex], sd_years)]


@dataclasses.dataclass(frozen=True)
class GrowthChartInput:
    sex: str
    bone_age: int
    chronological_age: Optional[int]


class GrowthChart:
    """
    Abstraction for a single-threaded growth chart generator.
    This is required since matplotlib is not thread-safe.
    """

    _input_queue: queue.Queue[GrowthChartInput]
    _output_queue: queue.Queue[np.ndarray]

    def __init__(self):
        self._input_queue = queue.Queue(1)
        self._output_queue = queue.Queue(1)
        self._running = True
        self._thread = threading.Thread(target=self._plot_thread)
        self._thread.start()

    def plot_chart(self, data: GrowthChartInput) -> np.ndarray:
        self._input_queue.put(data)
        result = self._output_queue.get()
        self._output_queue.task_done()
        return result

    def stop(self):
        self._running = False
        self._thread.join()

    def _plot_thread(self):
        while self._running:
            try:
                input = self._input_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            fig = plt.figure(figsize=(20, 20))
            fig.patch.set_facecolor('k')
            plt.plot(CHRONOLOGICAL_AGE[input.sex],
                     BONE_AGE[input.sex],
                     color = 'g', marker='o', markersize=14, linestyle = '--',
                     label = 'Brush Foundation Study')
            plt.plot(CHRONOLOGICAL_AGE[input.sex],
                     BOUND_LOW[input.sex],
                     color = 'b', marker='o', markersize=14, linestyle = '--',
                     label = 'Brush Foundation Study - lower bound')
            plt.plot(CHRONOLOGICAL_AGE[input.sex],
                     BOUND_HIGH[input.sex],
                     color = 'b', marker='o', markersize=14, linestyle = '--',
                     label = 'Brush Foundation Study - upper bound')

            if input.chronological_age is None:
                # draw a line if we don't have the patient's age
                plt.plot(np.arange(0, max(CHRONOLOGICAL_AGE[input.sex]) + 1),
                         [input.bone_age / 12] * (max(CHRONOLOGICAL_AGE[input.sex]) + 1),
                         color = 'r', linestyle = ':', linewidth = 4, label = 'prediction')
            else:
                # draw dot
                plt.plot(input.chronological_age / 12, input.bone_age / 12,
                         color = 'r', marker = 'D', label = 'prediction')
                # vertical line to the dot 
                plt.plot([0, input.chronological_age / 12],
                         [input.bone_age / 12] * 2,
                         color = 'w', linestyle = '--', linewidth = 2)
                # horizontal line to the dot 
                plt.plot([input.chronological_age / 12] * 2,
                         [0, input.bone_age / 12],
                         color = 'w', linestyle = '--', linewidth = 2)

            plt.grid(visible = True, which = 'both')
            plt.yticks()
            plt.legend(fontsize=20)

            ax = plt.gca()
            ax.set_xlabel('Chronological Age [years]', color = 'w', fontdict={'fontsize': 20})
            ax.set_xticks(np.arange(0, max(CHRONOLOGICAL_AGE[input.sex]) + 1))
            ax.tick_params(axis='x', colors = 'w', labelsize = 18)
            ax.set_ylabel('Bone Age [years]', color = 'w', fontdict={'fontsize': 20})
            ax.set_yticks(np.arange(0, math.ceil(max(BOUND_HIGH[input.sex])) + 1))
            ax.tick_params(axis='y', colors = 'w', labelsize = 18)
            ax.set_title('Bone Age Growth Chart', color = 'w', fontdict={'fontsize': 24})
            ax.set_facecolor('k')
            ax.grid(True, which='both')
            
            # https://stackoverflow.com/questions/67955433/how-to-get-matplotlib-plot-data-as-numpy-array
            canvas = ax.figure.canvas
            canvas.draw()
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            growth_chart = data.reshape(canvas.get_width_height()[::-1] + (3,))

            self._input_queue.task_done()
            self._output_queue.put(growth_chart)
