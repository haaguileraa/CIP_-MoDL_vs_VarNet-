import torch
import torch.nn as nn
import torch.nn.functional as F

epsilon = 1e-5
TFeps = torch.Tensor([1e-5])

def c2r(x):
    return torch.stack([x.real, x.imag], dim=-1)
def r2c(x):
    return torch.complex(x[..., 0], x[..., 1])

def __init__(self, szW, lastLayer):
    super(createLayer, self).__init__()
    self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(*szW)))
    self.lastLayer = lastLayer

def forward(self, x, training):
    x = F.conv2d(x, self.W, padding=(1, 1))
    xbn = nn.BatchNorm2d(x.size()[1], eps=epsilon, affine=False).to(x.device)
    xbn = xbn(x, training=training)

    if not self.lastLayer:
        return F.relu(xbn)
    else:
        return xbn

def __init__(self, nLay):
    super(Dw, self).__init__()
    self.nLay = nLay
    self.nw = nn.ParameterDict()
    self.szW = {key: (64, 2, 3, 3) for key in range(2, nLay)}
    self.szW[1] = (64, 2, 3, 3)
    self.szW[nLay] = (2, 64, 3, 3)

    for i in range(nLay):
        self.nw['c' + str(i)] = nn.Parameter(torch.zeros(1, 2, 1, 1))

    for i in range(1, nLay + 1):
        lastLayer = False
        if i == nLay:
            lastLayer = True
        setattr(self, 'layer'+str(i), createLayer(self.szW[i], lastLayer))

def forward(self, inp, training):
    with torch.no_grad():
        self.nw['c0'].data = inp
    shortcut = inp
    for i in range(1, self.nLay + 1):
        if i == self.nLay:
            lastLayer = True
        else:
            lastLayer = False
        nw_i = getattr(self, 'layer'+str(i))(self.nw['c'+str(i-1)], training)
        self.nw['c'+str(i)].data = nw_i
    dw = shortcut + self.nw['c'+str(self.nLay)]
    return dw


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, csm, mask, lam):
        self.nrow, self.ncol = mask.shape
        self.pixels = self.nrow * self.ncol
        self.mask = torch.from_numpy(mask).to(torch.bool)
        self.csm = torch.from_numpy(csm).to(torch.complex64)
        self.SF = torch.complex(torch.sqrt(torch.tensor(self.pixels, dtype=torch.float32)), 0.)
        self.lam = lam

    def myAtA(self, img):
        coilImages = self.csm * img
        kspace = torch.fft.fftn(coilImages) / self.SF
        temp = kspace * self.mask.to(kspace.device)
        coilImgs = torch.fft.ifftn(temp) * self.SF
        coilComb = torch.sum(coilImgs * torch.conj(self.csm), axis=0)
        coilComb = coilComb + self.lam * img
        return coilComb

def myCG(A, rhs):
    """
    This is my implementation of CG algorithm in PyTorch that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    rhs = rhs.to(torch.complex64)
    x = torch.zeros_like(rhs)
    r = rhs
    p = rhs
    rTr = torch.sum(r.conj() * r).real
    i = 0
    while i < 10 and rTr > 1e-10:
        Ap = A.myAtA(p)
        alpha = rTr / torch.sum(p.conj() * Ap).real
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj() * r).real
        beta = rTrNew / rTr
        p = r + beta * p
        rTr = rTrNew
        i += 1
    return x.real

def getLambda():
    """
    create a shared variable called lambda.
    """
    lam = torch.nn.Parameter(torch.tensor(.05), requires_grad=True)
    return lam

def callCG(rhs, csm, mask, lam):
    """
    this function will call the function myCG on each image in a batch
    """
    Aobj = Aclass(csm, mask, lam)
    rec = torch.stack([myCG(Aobj, r) for r in rhs], dim=0)
    return rec

def dcManualGradient(x):
    """
    This function impose data consistency constraint. Rather than relying on
    PyTorch to calculate the gradient for the conjuagte gradient part.
    We can calculate the gradient manually as well by using this function.
    Please see section III (c) in the paper.
    """
    y = callCG(x, csm, mask, lam1)
    out = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return y, out

@torch.custom_function
def dc(x, csm, mask, lam1):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    y = callCG(x, csm, mask, lam1)
    return y

def makeModel(atb, csm, mask, training, nLayers, K, gradientMethod):
    """
    This is the main function that creates the model.
    """
    out = {}
    out['dc0'] = atb
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        with torch.autograd.profiler.record_function("myModel"):
            for i in range(1, K + 1):
                j = str(i)
                out['dw' + j] = dw(out['dc' + str(i - 1)], training, nLayers)
                lam1 = getLambda()
                rhs = atb + lam1 * out['dw' + j]
                if gradientMethod == 'AG':
                    out['dc' + j] = dc(rhs, csm, mask, lam1)
                elif gradientMethod == 'MG':
                    if training:
                        out['dc' + j] = dcManualGradient(rhs)
                    else:
                        out['dc' + j] = dc(rhs, csm, mask, lam1)
    return out
