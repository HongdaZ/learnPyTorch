import torch
torch.cuda.is_available()
# autograd
x = torch.arange(4.0)
x.requires_grad_(True)
y = 2 * torch.dot(x, x)
y.backward()
x.grad
x.grad.zero_()
y = x * x
sum(y).backward()
x.grad