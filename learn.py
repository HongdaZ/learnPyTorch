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

x.grad.zero_()
y = x * x
u = y.detach()
z = x * u
sum(z).backward()
x.grad
x.grad.zero_()

y.sum().backward()
x.grad

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c =  1000 * b
    return c
a = torch.randn(size = (), requires_grad = True)
d= f(a)
d.backward()
a.grad == d / a

from torch.distributions import multinomial
import matplotlib.pyplot as plt

fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim = 0)
estimates = cum_counts / cum_counts.sum(dim = 1, keepdim = True)
estimates


for i in range(6):
    plt.plot(estimates[:, i].numpy(),
                 label = ("P(die = " + str(i + 1) + ")"))

plt.gca().set_xlabel("Groups of experiments")
plt.gca().set_ylabel("Estimated probability")
plt.legend()
plt.show(block = True)

