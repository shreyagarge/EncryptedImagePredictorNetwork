from __future__ import division

import numpy as np

def dotProduct(array1, array2):
    dotProduct = np.zeros((len(array1), len(array2[0])))
    
    if len(array1[0,:]) != len(array2[:,0]):
        return 0    
    print len(dotProduct[:,0]),len(dotProduct[0,:]),len(array1[0, :])
    for j in range(len(dotProduct[:,0])):
        for i in range(len(dotProduct[0, :])):
            sum = 0

            for column in range(len(array1[0, :])):
                    sum = sum + array1[j, column]*array2[column, i]

            dotProduct[j, i] = sum

    return dotProduct


def calc_pad(pad, in_siz, out_siz, stride, ksize):
    """Calculate padding width.

    Args:
        pad: padding method, "SAME", "VALID", or manually speicified.
        ksize: kernel size [I, J].

    Returns:
        pad_: Actual padding width.
    """
    if pad == 'SAME':
        return (out_siz - 1) * stride + ksize - in_siz
    elif pad == 'VALID':
        return 0
    else:
        return pad


def calc_size(h, kh, pad, sh):
    """Calculate output image size on one dimension.

    Args:
        h: input image size.
        kh: kernel size.
        pad: padding strategy.
        sh: stride.

    Returns:
        s: output size.
    """

    if pad == 'VALID':
        return np.ceil((h - kh + 1) / sh)
    elif pad == 'SAME':
        return np.ceil(h / sh)
    else:
        return int(np.ceil((h - kh + pad + 1) / sh))



def extract_sliding_windows(x, ksize, pad, stride, floor_first=True):
    """Converts a tensor to sliding windows.

    Args:
        x: [N, H, W, C]
        k: [KH, KW]
        pad: [PH, PW]
        stride: [SH, SW]

    Returns:
        y: [N, (H-KH+PH+1)/SH, (W-KW+PW+1)/SW, KH * KW, C]
    """
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]
    kh = ksize[0]
    kw = ksize[1]
    sh = stride[0]
    sw = stride[1]

    h2 = int(calc_size(h, kh, pad, sh))
    w2 = int(calc_size(w, kw, pad, sw))
    ph = int(calc_pad(pad, h, h2, sh, kh))
    pw = int(calc_pad(pad, w, w2, sw, kw))

    ph0 = int(np.floor(ph / 2))
    ph1 = int(np.ceil(ph / 2))
    pw0 = int(np.floor(pw / 2))
    pw1 = int(np.ceil(pw / 2))

    if floor_first:
        pph = (ph0, ph1)
        ppw = (pw0, pw1)
    else:
        pph = (ph1, ph0)
        ppw = (pw1, pw0)
    x = np.pad(
        x, ((0, 0), pph, ppw, (0, 0)),
        mode='constant',
        constant_values=(0.0, ))

    y = np.zeros([n, h2, w2, kh, kw, c])
    for ii in range(h2):
        for jj in range(w2):
            xx = ii * sh
            yy = jj * sw
            y[:, ii, jj, :, :, :] = x[:, xx:xx + kh, yy:yy + kw, :]
    return y


def conv2d(x, w, pad='SAME', stride=(1, 1)):
    """2D convolution (technically speaking, correlation).
    
    takes (1,height,width,in_channels) image and (height,width,in_channels,out_channels) kernel 
    and gives (1,height,width,out_channels) output
    
    Args:
        x: [N, H, W, C]
        w: [I, J, C, K]
        pad: [PH, PW]
        stride: [SH, SW]

    Returns:
        y: [N, H', W', K]
    """
    ksize = w.shape[:2]
    x = extract_sliding_windows(x, ksize, pad, stride)
    ws = w.shape
    w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
    xs = x.shape
    x = x.reshape([xs[0] * xs[1] * xs[2], -1])
    y = dotProduct(x,w)
    y = y.reshape([xs[0], xs[1], xs[2], -1])
    return y


def add_bias(a,b):
    """Bias addition after convolution -  bias to be added to the entire matrix in each channel
    
    Args:
        a: [1,height,width,channels] 
        b: [channels]

    Returns:
        a: [1,height,width,channels]
    """
    for i in range(a.shape[3]):
        a[0,:,:,i] += b[i]

    return a

def add_bias1(x,y):
    """bias addition after fc - only one channel but different bias added to each node(element in the array)

    Args:
        x: [nodes]
        y: [nodes]

    Returns:
        x: [nodes]
    """
    for i in range(x.shape[0]):
        x[i] = x[i] + y[i]
    return x


# def mean(x):
#     mean1 = np.zeros(1,(int(x.shape[1]/2),int(x.shape[2]/2)),x.shape[3])
#     for i in range(x.shape[3]): 
#         mean1[0,:,:,i] = meanpool(con1[0,:,:,i])

#     return mean1

# def meanpool(x):
#     resmat = np.zeros((int(x.shape[0]/2),int(x.shape[1]/2)))
#     ii,jj,i,j=0,0,0,0
#     while i < x.shape[0]:
#         j,jj=0,0
#         while j < x.shape[1]:
#             resmat[ii,jj] = (x[i,j]+x[i+1,j]+x[i,j+1]+x[i+1,j+1])*0.25
#             jj+=1
#             j+=2
#         ii+=1
#         i+=2
#     return resmat

def meanpool2(x):
    """meanpool2 takes (1,height,width,channels) input and performs meanpooling on each of the #channels
       matrices seperately and gives a (1,height/2,width/2,channels) output

    Args:
        x: [1,n,h,c]

    Returns:
        y: [1,n/2,h/2,c]
    """
    retval = np.zeros((1,int(x.shape[1]/2),int(x.shape[2]/2),x.shape[3]))
    for chan in range(x.shape[3]):
        ii,jj,i,j=0,0,0,0
        while i < x.shape[1]:
            j,jj=0,0
            while j < x.shape[2]:
                retval[0,ii,jj,chan] = (x[0,i,j,chan]+x[0,i+1,j,chan]+x[0,i,j+1,chan]+x[0,i+1,j+1,chan])*0.25
                jj+=1
                j+=2
            ii+=1
            i+=2

    return retval


def fconnect(x,y):
    retval = np.zeros((y.shape[1]))
    for i in range(y.shape[1]):
        res = 0
        for j in range(y.shape[0]):
            res += x[j]*y[j,i]
        retval[i] = res
    return retval


def activ_fun(x):
    if len(x.shape) == 1:
        s1 = x.shape[0]
        squared = np.zeros((s1))
        for i in range(s1):
            squared[i] = x[i]*x[i]
    else:
        s1 = x.shape[1]
        s2 = x.shape[2]
        s3 = x.shape[3]
        squared = np.zeros((1,s1,s2,s3))
        for i in range(s1):
            for j in range(s2):
                for k in range(s3):
                    squared[0,i,j,k] = x[0,i,j,k]*x[0,i,j,k]
    return squared
