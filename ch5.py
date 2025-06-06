import torch
from torch import nn
from torch.nn import functional as F
net = nn.Sequential(nn.Linear(20, 256),
                    nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
net(X)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

net = MLP()
net(X)

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
net = MySequential(nn.Linear(20, 256),
                   nn.ReLU(),
                   nn.Linear(256, 10))
net(X)

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad = False)
        self.linear = nn.Linear(20, 20)
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
net(X)

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20),
                        FixedHiddenMLP())
chimera(X)

net = nn.Sequential(nn.Linear(4, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1))
X = torch.rand(size = (2, 4))

net(X)

print(net[2].state_dict())

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
net[2].weight.grad == None

net.state_dict()["2.bias"].data

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f"block {i}", block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(rgnet)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std = .01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)

net[0].weight.data[0], net[0].bias.data[0]

def init_xvaxier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(init_xvaxier)
net[2].apply(init_42)

net[0].weight.data[0]
net[2].weight.data

def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >=5
net.apply(my_init)
net[0].weight[:2]

shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)

print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
net = nn.Sequential(nn.Linear(8, 128),
                    CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
linear.weight
linear(torch.rand(2, 5))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))

x = torch.arange(4)
torch.save(x, "x-file")
x2 = torch.load("x-file")
x2
y = torch.zeros(4)
mydict = {"x" : x, "y" : y}
torch.save(mydict, "mydict")
mydict2 = torch.load("mydict")
mydict2

net = MLP()
X = torch.randn(size = (2, 20))
Y = net(X)

torch.save(net.state_dict(), "mlp.params")
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()
Y_clone = clone(X)
Y_clone == Y

torch.device("cpu"), torch.device("cuda")
def try_gpu(i = 0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")

def try_all_gpus():
    devices = [torch.device(f"cuda:{i}")
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]

try_gpu(), try_all_gpus(), try_all_gpus()
X.device
X = torch.ones(2, 3, device = try_gpu())
X.device
Z = X.cpu()

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device = try_gpu())
net(X)
net[0].weight.data.device

