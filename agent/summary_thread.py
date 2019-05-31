import torch.multiprocessing as mp

from tensorboardX import SummaryWriter
class SummaryThread(mp.Process):
    def __init__(self,
                input_queue : mp.Queue):
        super(SummaryThread, self).__init__()
        self.i_queue = input_queue
        self.exit = mp.Event()

    def run(self):
        print("SummaryThread starting")
        self.writer = SummaryWriter("runs/log_output/run")
        while True and not self.exit.is_set():
            name, scalar, step = self.i_queue.get()
            self.writer.add_scalar(name, scalar, step)
    def stop(self):
        print("Stop initiated for GPUThread")
        self.exit.set()
