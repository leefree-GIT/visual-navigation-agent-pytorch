import torch.multiprocessing as mp
import torch.nn as nn
import torch
import queue
import logging
import h5py
import torchvision.transforms.functional as F
from torchvision import transforms

class GPUThread(mp.Process):
    def __init__(self,
                model : torch.nn.Module,
                device : torch.device,
                input_queues : mp.Queue,
                output_queues : mp.Queue,
                scenes,
                evt):
        super(GPUThread, self).__init__()
        self.model = model
        self.device = device
        self.i_queues = input_queues
        self.o_queues = output_queues
        self.exit = mp.Event()
        self.scenes = scenes 
        self.evt = evt
        
    def run(self):
        self.model = self.model.to(self.device)
        print("GPUThread starting")
        while True and not self.exit.is_set():
            self.evt.wait()
            for ind, i_q in enumerate(self.i_queues):
                try:
                    frame = i_q.get(False)
                    tensor = frame.to(self.device)
                    output_tensor = self.model((tensor,))
                    output_tensor = output_tensor.permute(1,0)
                    output_tensor = output_tensor.cpu()
                    self.o_queues[ind].put(output_tensor)
            

                except queue.Empty as e:
                    pass
            self.evt.clear()
    def stop(self):
        print("Stop initiated for GPUThread")
        self.exit.set()
        self.evt.set()
