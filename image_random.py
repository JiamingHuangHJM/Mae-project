from utils import take_indexes
import numpy as np
import torch


class ImageRandom(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def __random_indexes(size : int):
        new_idx = np.arange(size)
        np.random.shuffle(new_idx)
        return new_idx, np.argsort(new_idx)


    def forward(self, pths):
        indexes = []
        for _ in range(pths.shape[1]):
            indexes.append(self.__random_indexes(pths.shape[0]))
        
        f_idx = []
        b_idx = []
        for i in indexes:
            f_idx.append(i[0])
        
        for i in indexes:
            b_idx.append(i[0])

        f_idx = torch.as_tensor(np.stack(f_idx, axis=-1), dtype=torch.long)
        b_idx = torch.as_tensor(np.stack(b_idx, axis=-1), dtype=torch.long)

        f_idx = f_idx.to(pths.device)
        b_idx = b_idx.to(pths.device)

        pths = take_indexes(pths, f_idx)
        pths = pths[:int(pths.shape[0] * (1 - self.ratio))]

        return pths, f_idx, b_idx
