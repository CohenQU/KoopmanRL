from torch import nn
import torch

def gaussian_init_(nodes_in_layer1, nodes_in_layer2, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/nodes_in_layer2]))
    Omega = sampler.sample((nodes_in_layer2, nodes_in_layer1))[..., 0]  
    print("Omega.shape = ", Omega.shape)
    return Omega


class encoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, b)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))        
        x = self.fc3(x)
        
        return x


class decoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(decoderNet, self).__init__()

        self.m = m
        self.n = n
        self.b = b

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(b, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, m*n)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x)) 
        x = self.tanh(self.fc2(x)) 
        x = self.tanh(self.fc3(x))
        x = x.view(-1, 1, self.m, self.n)
        return x



class dynamics(nn.Module):
    def __init__(self, b, action_dim, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b + action_dim, b, bias=False)
        #self.dynamics.weight.data = gaussian_init_(b + action_dim, b, std=1)           
        #U, _, V = torch.svd(self.dynamics.weight.data)
        #print("U.shape = ", U.shape)
        #print("V.shape = ", V.shape)
        #self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale
        
    def forward(self, x):
        x = self.dynamics(x)
        return x


class dynamics_back(nn.Module):
    def __init__(self, b, action_dim, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b+action_dim, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x


class koopmanAE(nn.Module):
    def __init__(self, state_dim, action_dim, b, steps, steps_back, alpha = 1, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.encoder = encoderNet(state_dim, 1, b, ALPHA = alpha)
        self.dynamics = dynamics(b, action_dim, init_scale)
        #self.backdynamics = dynamics_back(b, action_dim, self.dynamics)
        self.decoder = decoderNet(state_dim, 1, b, ALPHA = alpha)


    def forward(self, state, actions, mode='forward'):
        out = []
        out_back = []
        
        z = self.encoder(state.contiguous())
        q = z.contiguous()

        if mode == 'forward':
            for i in range(self.steps):
                a = actions[i].reshape(-1, 1, self.action_dim)
                #print("a.shape = ", a.shape)
                q = torch.cat((q, a), dim=2)
                #print("q.shape = ", q.shape)
                q = self.dynamics(q)
                #print("q.shape = ", q.shape)
                #print("self.decoder(q).shape = ", self.decoder(q).shape)
                out.append(self.decoder(q))

            q = z.contiguous()
            out.append(self.decoder(q))
            return out, out_back

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back
