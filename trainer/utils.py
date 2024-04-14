# torch
import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

class Transformer_2D(nn.Module):
    def __init__(self):
        super(Transformer_2D, self).__init__()
    # @staticmethod
    def forward(self,src, flow):
        b = flow.shape[0]
        h = flow.shape[2]
        w = flow.shape[3]

        size = (h,w)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b,1,1,1).cuda()
        new_locs = grid+flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1 , 0]]
        warped = F.grid_sample(src,new_locs,align_corners=True,padding_mode="border")
        # ctx.save_for_backward(src,flow)
        return warped


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class Resize():
    def __init__(self, size_tuple, use_cv=True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv

    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=[self.size_tuple[0], self.size_tuple[1]])
        tensor = tensor.squeeze(0)

        return tensor  # 1, 64, 128, 128


class ToTensor():
    def __call__(self, tensor):
        """
        tensor: H W C
        target: C H W
        """
        if len(tensor.shape) == 2:
            tensor = np.expand_dims(tensor, 0)
            # tensor = np.array(tensor)
            return torch.from_numpy(tensor)  # C H W
        elif len(tensor.shape) == 3:
            return torch.from_numpy(tensor.transpose(2, 0, 1))  # C H W
