import numpy as np
import os
import sys

class params_RatMap:
    def init(self):
        self.Npts = 100

def LOCAL_SetupDefaults():

    S = params_RatMap()
    # Calcualtion defaults
    S.Npts = 100
    S.TGL_PHI = 'proj'

    # Setup ellipsoid defaults
    S.E_C = [1.9128, -7.6549, -5.4439]
    S.E_R = [9.5304, 5.5393, 6.9745]
    S.E_OA = [106.5100, -2.5211, -19.5401]

    # Setup whisker transformation parameters and equations
    # (parameters must be None to use equations)

    S.EQ_BP_th = [15.2953, 0, -144.2220]
    S.BP_th = None

    S.EQ_BP_phi = [0, 18.2237, 34.7558]
    S.BP_phi = None

    S.EQ_W_s = [-7.9312,2.2224,52.1110]
    S.W_s = None

    S.EQ_W_a = [-0.02052,0,-0.2045]
    S.W_a = None

    S.EQ_W_th = [10.6475,0,37.3178]
    S.W_th = None

    S.EQ_W_psi = [18.5149,49.3499,-50.5406]
    S.W_psi = None

    S.EQ_W_zeta = [18.7700,-11.3485,-4.9844]
    S.W_zeta = None

    S.EQ_W_phi = [1.0988,-18.0334,50.6005]
    S.W_phi = None
    S.EQ_W_phiE = [0,-15.8761,47.3263]
    S.W_phiE = None

    return S

def LOCAL_SetupWhiskerNames(wselect):

    wname = ['A0','A1','A2','A3','A4',
              'B0','B1','B2','B3','B4','B5',
              'C0','C1','C2','C3','C4','C5','C6',
              'D0','D1','D2','D3','D4','D5','D6',
              'E1','E2','E3','E4','E5','E6']

    lwname  = ['L' + nam_per for nam_per in wname]
    rwname  = ['R' + nam_per for nam_per in wname]
    lwname.extend(rwname)
    wname   = lwname
    allow_list  = "LRABCDE0123456"
    for wpart in wselect:
        if wpart not in allow_list:
            continue
        wname   = [wname_tmp for wname_tmp in wname if wpart in wname_tmp]

    return wname

if __name__=="__main__":
    S   = LOCAL_SetupDefaults()
    wselect = []
    S.wname = LOCAL_SetupWhiskerNames(wselect)

    #print(S.EQ_W_phiE)
    #print(S.wname)
