import numpy as np

def apply_distortion(x,k):

    len_k = len(k)
    if len_k < 5:
        for indx in xrange(len(k), 5):
            k.append(0)

    m,n = x.shape
    
    #print(x[0, 99])
    #print(x[1, 99])
    r2 = np.square(x[0,:]) + np.square(x[1,:]);
    #print(r2[99])
    r4 = np.square(r2)
    r6 = np.power(r2, 3)

    cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6
    dcdistdk = np.asarray([r2.transpose(), r4.transpose(), np.zeros(n), np.zeros(n), r6.transpose()]).transpose()
    #print(dcdistdk.shape)
    #print(dcdistdk[99, 1])
    #print(r4[99])

    xd1 = x * np.dot(np.ones([2,1]), cdist[np.newaxis, :])

    coeff = np.concatenate([cdist,cdist]).reshape([2*n,1])*np.ones([1,3])
    
    dxd1dk = np.zeros([2*n,5]);
    dxd1dk[0::2,:] = (x[0,:].transpose()[:, np.newaxis]*np.ones((1,5)))*dcdistdk;
    #print(dxd1dk[0, 0])
    #print(dxd1dk[2, 0])
    dxd1dk[1::2,:] = (x[1,:].transpose()[:, np.newaxis]*np.ones((1,5)))*dcdistdk;
    #print(dxd1dk[1, 0])
    #print(dxd1dk[3, 0])

    a1 = 2*x[0,:]*x[1,:];
    a2 = r2 + 2*np.square(x[0,:]);
    a3 = r2 + 2*np.square(x[1,:]);

    delta_x = np.asarray([k[2]*a1 + k[3]*a2, k[2]*a3 + k[3]*a1])
    #print(delta_x[0, 99])

    aa = (2*k[2]*x[1,:]+6*k[3]*x[0,:]).transpose()[:, np.newaxis]*np.ones((1,3))
    bb = (2*k[2]*x[0,:]+2*k[3]*x[1,:]).transpose()[:, np.newaxis]*np.ones((1,3))
    cc = (6*k[2]*x[1,:]+2*k[3]*x[0,:]).transpose()[:, np.newaxis]*np.ones((1,3))

    ddelta_xdk = np.zeros((2*n,5))
    ddelta_xdk[0::2,2] = a1.transpose()
    ddelta_xdk[0::2,3] = a2.transpose()
    ddelta_xdk[1::2,2] = a3.transpose()
    ddelta_xdk[1::2,3] = a1.transpose()

    #print(xd1[:, 479:500])
    #print(delta_x[:, 479:500])

    xd = xd1 + delta_x
    dxddk = dxd1dk + ddelta_xdk
    if len_k < 5:
        dxddk = dxddk[:,:len_k]

    return xd,dxddk
