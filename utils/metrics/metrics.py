import torch

class CMD(torch.nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self, n_moments=5):
        super(CMD, self).__init__()
        self.n_moments = n_moments

    def forward(self, x1, x2):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(self.n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
    
    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
    

class CD(torch.nn.Module):

    def __init__(self):
        super(CD, self).__init__()

    def forward(self, x1, x2):
        mean_x1 = x1.mean(axis=0)
        mean_x2 = x2.mean(axis=0)

        diff = mean_x1 - mean_x2

        return diff.norm()