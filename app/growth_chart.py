import dataclasses
import queue
import threading

import matplotlib.pyplot as plt
import numpy as np
import scipy


CHRONOLOGICAL_AGE = {
    'M': [3/12, 6/12, 9/12, 12/12, 18/12, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    'F': [3/12, 6/12, 9/12, 12/12, 18/12, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
}
BONE_AGE = {
    'M': [3.01, 6.09, 9.56, 12.74, 19.36, 25.97, 32.40, 38.21, 43.89, 49.04, 56.00, 62.43, 75.46, 88.20, 101.38, 113.90, 125.68, 137.32, 148.82, 158.39, 170.02, 182.72, 195.32, 206.21],
    'F': [3.02, 6.04, 9.05, 12.04, 18.22, 24.16, 30.96, 36.63, 43.50, 50.14, 60.06, 66.21, 78.50, 89.30, 100.66, 113.86, 125.66, 137.86, 149.62, 162.28, 174.25, 183.62, 189.44]
}


@dataclasses.dataclass(frozen=True)
class GrowthChartInput:
    sex: str
    bone_age: int


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
            
            fig = plt.figure(figsize=(20, 10))
            plt.plot(CHRONOLOGICAL_AGE[input.sex],
                     BONE_AGE[input.sex],
                     color = 'g', marker='o', linestyle = '--',
                     label = 'Brush Foundation Study')
            f = scipy.interpolate.interp1d(BONE_AGE[input.sex], CHRONOLOGICAL_AGE[input.sex])
            pred_int_x = f(input.bone_age)
            plt.plot(pred_int_x, input.bone_age,
                     color = 'r', marker = 'D', label = 'prediction')
            # plt.plot(np.arange(0, max(chronological_age)+1), [pred_int] * (max(chronological_age)+1), color = 'b', linestyle = ':', label = 'prediction')
            plt.xlabel('Chronological Age [years]')
            plt.ylabel('Bone Age [months]')
            plt.title('Bone Age Growth Chart')
            plt.grid(visible = True, which = 'both')
            plt.legend()

            # https://stackoverflow.com/questions/67955433/how-to-get-matplotlib-plot-data-as-numpy-array
            canvas = plt.gca().figure.canvas
            canvas.draw()
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            growth_chart = data.reshape(canvas.get_width_height()[::-1] + (3,))

            self._input_queue.task_done()
            self._output_queue.put(growth_chart)
