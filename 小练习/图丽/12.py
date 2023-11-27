import multiprocessing


class ProcessManager:
    def __init__(self):
        self.process = None

    def _worker(self, *args, **kwargs):
        # This function will run in a separate process
        pass

    def start_process(self, *args, **kwargs):
        if self.process is None or not self.process.is_alive():
            self.process = multiprocessing.Process(
                target=self._worker, args=args, kwargs=kwargs
            )
            self.process.start()

    def stop_process(self):
        if self.process:
            self.process.terminate()
            self.process.join()
