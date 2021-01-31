import numpy as np

def titaphi2lambdai(tita,phi):
    lamb_rd = np.arcsin(-(np.cos(phi*np.pi/180.0)*np.cos(tita*np.pi/180.0)))
    lamb = lamb_rd*180.0/np.pi

    i_rd= np.arctan2(np.tan(tita*np.pi/180.0),np.sin(phi*np.pi/180.0))
    i = i_rd*180.0/np.pi
    return lamb, i
