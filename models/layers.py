import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        assert 0 <= p < 1, "p must be in [0, 1)"
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        
        mask = (torch.rand(x.shape, device=x.device) > self.p).float()
        return x * mask / (1 - self.p)
    
    def extra_repr(self):
        return f'p={self.p}'

drop = CustomDropout(p=0.5)

# test training mode
drop.train()
x = torch.ones(1000)
out = drop(x)
print(out.sum())  # should be roughly 1000 (scaled), ~500 zeros

# test eval mode
drop.eval()
out = drop(x)
print(out.sum())  # should be exactly 1000.0 (no dropout)