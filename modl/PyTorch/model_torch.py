"""
This code will create the model described in our following paper
MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

Paper dwonload  Link:     https://arxiv.org/abs/1712.02862

@author: haggarwal
"""

import torch
import numpy as np
from os.path import expanduser
home = expanduser("~")
epsilon=1e-5
TFeps=torch.tensor(1e-5,dtype=torch.float32)

# function c2r contatenate complex input as new axis two two real inputs
c2r=lambda x:torch.stack([torch.real(x),torch.imag(x)],dim=-1)
#r2c takes the last dimension of real input and converts to complex
r2c=lambda x:torch.complex(x[...,:1],x[...,1:])

def createLayer(x, szW, trainning,lastLayer):
    """
    This function create a layer of CNN consisting of convolution, batch-norm,
    and ReLU. Last layer does not have ReLU to avoid truncating the negative
    part of the learned noise and alias patterns.
    """
    W=torch.nn.Parameter(torch.randn(szW,dtype=torch.float32)*0.01)
    x = torch.nn.functional.conv2d(input=x, weight=W, stride=[1, 1, 1, 1], padding='SAME')
    xbn=torch.nn.BatchNorm2d(x.shape[1],affine=True,track_running_stats=True)
    xbn.train(trainning)
    xbn=xbn(x)
    if not(lastLayer):
        return torch.nn.functional.relu(xbn)
    else:
        return xbn

def dw(inp,trainning,nLay):
    """
    This is the Dw block as defined in the Fig. 1 of the MoDL paper
    It creates an n-layer (nLay) residual learning CNN.
    Convolution filters are of size 3x3 and 64 such filters are there.
    nw: It is the learned noise
    dw: it is the output of residual learning after adding the input back.
    """
    lastLayer=False
    nw={}
    nw['c'+str(0)]=inp
    szW={}
    szW = {key: (3,3,64,64) for key in range(2,nLay)}
    szW[1]=(3,3,2,64)
    szW[nLay]=(3,3,64,2)

    for i in np.arange(1,nLay+1):
        if i==nLay:
            lastLayer=True
        with torch.no_grad():
            nw['c'+str(i)]=createLayer(nw['c'+str(i-1)],szW[i],trainning,lastLayer)

    with torch.no_grad():
        shortcut=torch.identity(inp)
        dw=shortcut+nw['c'+str(nLay)]
    return dw


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, csm,mask,lam):
        self.nrow,self.ncol=mask.shape[0],mask.shape[1]
        self.pixels=self.nrow*self.ncol
        self.mask=mask
        self.csm=csm
        self.SF=np.sqrt(self.pixels)
        self.lam=lam
        #self.cgIter=cgIter
        #self.tol=tol
    def myAtA(self,img):
        with torch.no_grad():
            coilImages=self.csm*img
            kspace=  torch.fft(coilImages)/self.SF
            temp=kspace*self.mask
            coilImgs =torch.ifft(temp)*self.SF
            coilComb= torch.sum(coilImgs*torch.conj(self.csm),dim=0)
            coilComb=coilComb+self.lam*img
        return coilComb

def myCG(A,rhs):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    rhs=r2c(rhs)
    cond=lambda i,rTr,*_: torch.logical_and( torch.less(i,10), rTr>1e-10)
    def body(i,rTr,x,r,p):
        with torch.no_grad():
            Ap=A.myAtA(p)
            alpha = rTr / torch.cast(torch.reduce_sum(torch.conj(p)*Ap), dtype=torch.float32)
            alpha=torch.complex(alpha,0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = torch.cast( torch.reduce_sum(torch.conj(r)*r), dtype=torch.float32)
            beta = rTrNew / rTr
            beta=torch.complex(beta,0.)
            p = r + beta * p
        return i+1,rTrNew,x,r,p

    x=torch.zeros_like(rhs)
    i,r,p=0,rhs,rhs
    rTr = torch.cast( torch.reduce_sum(torch.conj(r)*r), dtype=torch.float32,)
    loopVar=i,rTr,x,r,p
    out=torch.while_loop(cond=cond,body=body,loop_vars=loopVar,name='CGwhile',parallel_iterations=1)[2]
    return c2r(out)

def getLambda():
    """
    create a shared variable called lambda
    """
    lam = torch.nn.Parameter(torch.tensor(0.1,dtype=torch.float32))
    return lam

def callCG(rhs):
    """
    this function will call the function myCG on each image in a batch
    """
    G=torch.get_default_graph()
    getnext=G.get_operation_by_name('getNext')
    _,_,csm,mask=getnext.outputs
    l=getLambda()
    l2=torch.complex(l,0.)
    def fn(tmp):
        c,m,r=tmp
        Aobj=Aclass(c,m,l2)
        y=myCG(Aobj,r)
        return y
    inp=(csm,mask,rhs)
    rec=torch.map_fn(fn,inp,dtype=torch.float32,name='mapFn2' )
    return rec


class dcManualGradient(torch.autograd.Function):
    """
    This function impose data consistency constraint. Rather than relying on
    TensorFlow to calculate the gradient for the conjuagte gradient part.
    We can calculate the gradient manually as well by using this function.
    Please see section III (c) in the paper.
    """
    @staticmethod
    def forward(ctx, x):
        y=callCG(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        out=callCG(grad_output)
        return out

def dc(rhs,csm,mask,lam1):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    lam2=torch.complex(lam1,0.)
    def fn( tmp ):
        c,m,r=tmp
        Aobj=Aclass( c,m,lam2 )
        y=myCG(Aobj,r)
        return y
    inp=(csm,mask,rhs)
    rec=torch.map_fn(fn,inp,dtype=torch.float32,name='mapFn' )
    return rec

def makeModel(atb,csm,mask,training,nLayers,K,gradientMethod):
    """
    This is the main function that creates the model.

    """
    out={}
    out['dc0']=atb
    with torch.name_scope('myModel'):
        with torch.variable_scope('Wts',reuse=tf.AUTO_REUSE):
            for i in range(1,K+1):
                j=str(i)
                out['dw'+j]=dw(out['dc'+str(i-1)],training,nLayers)
                lam1=getLambda()
                rhs=atb + lam1*out['dw'+j]
                if gradientMethod=='AG':
                    out['dc'+j]=dc(rhs,csm,mask,lam1)
                elif gradientMethod=='MG':
                    if training:
                        out['dc'+j]=dcManualGradient(rhs)
                    else:
                        out['dc'+j]=dc(rhs,csm,mask,lam1)
    return out
