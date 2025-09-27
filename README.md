import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from first_net import *
from sampler import Sampler

import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
torch.set_default_tensor_type(torch.FloatTensor)

class Block(nn.Module):


    def __init__(self, in_N, width, out_N):
        super(Block, self).__init__()
        # create the necessary linear layers
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        # choose appropriate activation function
        self.phi =nn.ReLU()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x))))

class Net_1(nn.Module):
    """
    drrnn -- Deep Ritz Residual Neural Network

    Implements a network with the architecture used in the
    deep ritz method paper

    Parameters:
        in_N  -- input dimension
        out_N -- output dimension
        m     -- width of layers that form blocks
        depth -- number of blocks to be stacked
        phi   -- the activation function
    """

    def __init__(self, in_N, m, out_N, depth):
        super(Net_1, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.ReLU

        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N, m))

        # add middle blocks to list
        for i in range(depth):
            self.stack.append(Block(m, m, m))

        # add output linear layer
        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.001)
# Time limits
class Net_2(nn.Module):


    def __init__(self, in_N, m, out_N, depth):
        super(Net_2, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.Tanh()

        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N, m))

        # add middle blocks to list
        for i in range(depth):
            self.stack.append(Block(m, m, m))

        # add output linear layer
        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x



T0 = 0.0 + 1e-10    # Initial time
T = 1.0         # Terminal time

# Space limits
S_1 = 0.0 + 1e-10    # Low boundary
S_2 = 1.0            # High boundary



# initial condition
def g(S): return np.sin(10*S[:, 1])


s1 = Sampler([T0,T, S_1, S_2],
             [ 0,0,0, 0])
s2 = Sampler([ T0,T,S_1, S_2],
             [ 0, 0,0,0])
s3 = Sampler([T0,T,S_2, S_2],
             [0, 0,0,0])
s4 = Sampler([T0, T0, S_1, S_2],
             [0, 0, 0, 0])
s5 = Sampler([T, T, S_1, S_2],
             [0, 0, 0, 0])

def U(x,y):
    return x+y

def Loss(model,S1,S2):

    L1=torch.norm( model(S1).T+torch.exp(S1[:, 0])*torch.mean(torch.exp(S2[:, 0])*model(S2).T)
                   -0.5*(2.71828+1)*torch.exp(S1[:, 0])-S1[:, 0]-S1[:, 1],2)

    return L1/1000

model = Net_1(2,  20 ,1,1)
model.apply(weights_init)
opt = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.99)

# Number of samples
NS_1 = 10000
NS_2 = 10000


# Training parameters
steps_per_sample = 50
sampling_stages = 500


def train_model(model, optimizer, scheduler, num_epochs):
    sample1 = torch.tensor(s1.get_sample(NS_1), requires_grad=True)

    sample2 = torch.tensor(s2.get_sample(NS_2), requires_grad=True)
    for epoch in range(num_epochs):


        since = time.time()
        model.train()


        scheduler.step()

        for _ in range(steps_per_sample):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            L1 = Loss(model,sample1,sample2)

            loss = L1
            # backward + optimize
            #print_graph(loss.grad_fn,0)
            loss.backward()
            optimizer.step()

        epoch += 1
        if epoch % (num_epochs//num_epochs) == 0: print(f'epoch {epoch}, loss {loss.data}, L1 : {L1.data}')
    time_elapsed = time.time() - since
    print(f"Training finished in {time_elapsed:.2f} for {num_epochs}.")
    print(f"The final loss value is {loss.data}")

    if not os.path.exists('Aheckpoints'):
        os.mkdir('Aheckpoints')
    torch.save({
        'state_dict': model.state_dict(),
    }, 'Aheckpoints/aheckpoint.tar')

    return model

if os.path.exists('Aheckpoints/aheckpoint.tar'):
    checkpoint = torch.load("Aheckpoints/aheckpoint.tar")
    model.load_state_dict(checkpoint['state_dict'])

else:
    model= train_model(model, opt, scheduler, sampling_stages)
ax = plt.axes(projection='3d')

A=np.linspace(0,1,100)
#A=np.linspace(0,1,300)
# x=np.zeros((100, 6))
#output=np.zeros((300, 1))
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval()


output=np.zeros((100, 100))
B=np.linspace(0,1,100)

model.eval()
for j in range(len(A)):
    for i in range(len(A)):
        x=[A[j], A[i]]
        X = torch.tensor(x, dtype=torch.float32)
        output[j, i] = torch.tensor(model(X))#-U(A[j],A[i])**2
print(output)
A, B = np.meshgrid(A, B)
ax.plot_surface(A, B,output, cmap=plt.cm.winter)
plt.show()
