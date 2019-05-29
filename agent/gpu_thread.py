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
                h5_file_path,
                evt):
        super(GPUThread, self).__init__()
        self.model = model
        self.device = device
        self.i_queues = input_queues
        self.o_queues = output_queues
        self.exit = mp.Event()
        self.scenes = scenes 
        self.h5_file_path = h5_file_path
        self.evt = evt


    def _preprocess_obs(self, scenes, h5_file_path):
        device = torch.device('cuda')
        h5_path = lambda scene: h5_file_path.replace('{scene}', scene)
        self.d = dict()



        for scene in scenes:
            h5_file = h5py.File(h5_path(scene), 'r')
            if not scene in self.d:
                obs = h5_file['observation'][0]
                # resized = resize(obs, (224,224)).astype(dtype=np.float32)
                resized_tens_cat = torch.from_numpy(obs)
                resized_tens_cat = transform(F.to_pil_image(resized_tens_cat))
                resized_tens_cat = resized_tens_cat.unsqueeze(0)

                for i in range(1,len(h5_file['observation'])):
                    obs = h5_file['observation'][i]
                    resized_tens = torch.from_numpy(obs)
                    resized_tens = transform(F.to_pil_image(resized_tens))
                    resized_tens = resized_tens.unsqueeze(0)
                    resized_tens = resized_tens.share_memory_()
                    resized_tens_cat = torch.cat((resized_tens_cat, resized_tens), 0)
                resized_tens_cat = resized_tens_cat.share_memory_()
                resized_tens_cat = resized_tens_cat.to(device)

                logging.info(f"Tensor for scene {scene} created")
                size_v = resized_tens_cat.element_size() * resized_tens_cat.nelement()
                logging.info(f"{scene} size = {size_v}.")
                self.d[scene] = resized_tens_cat
    def run(self):
        # self._preprocess_obs(self.scenes, self.h5_file_path)
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
                    # self.evt.wait()
                    # scene, state = i_q.get(False)
                    # res_obs_scene = self.d[scene][state]
                    # tensor = res_obs_scene.to(self.device)
                    # tensor = tensor.unsqueeze(0)
                    # output_tensor = self.model((tensor,))
                    # output_tensor = output_tensor.permute(1,0)
                    # output_tensor = output_tensor.cpu()
                    # self.o_queues[ind].put(output_tensor)

                except queue.Empty as e:
                    pass
    def stop(self):
        print("Stop initiated for GPUThread")
        self.exit.set()
