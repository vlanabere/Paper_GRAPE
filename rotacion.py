## Rotacion de nubes magneticas
## Input coordenadas en GSE-Cordinates
## Output corrdenadas en MC-Cordinates
import numpy as np

def consistencia(R):
    R_transpuesta = np.transpose(R)
    M=np.dot(R,R_transpuesta)
    n, m =np.shape(M)
    epsilon=0.000001
    for i in range(len(M)):
        for j in range(len(M[0])):
            if abs(M[i][j])<epsilon:
                M[i][j]=0
            elif abs(M[i][j]-1)<epsilon:
                M[i][j]=1
    I=np.identity(n)
    return (M==I).all()

def norma_vector(mc_cordinates):
    norma = (mc_cordinates[0]**2+mc_cordinates[1]**2+mc_cordinates[2]**2)**0.5
    epsilon=0.000001
    return norma

def calculo_gamma(phi, tita):
    '''
    x_MC dot x_gse >0
    '''
    epsilon = 10e-15
    gamma = np.arctan(-np.tan(phi)/np.sin(tita+epsilon))
    xmc_dot_xgse = np.cos(gamma)*np.sin(tita)*np.cos(phi)-np.sin(gamma)*np.sin(phi)
    if xmc_dot_xgse < 0:
        gamma = np.arctan(-np.tan(phi)/np.sin(tita+epsilon))+np.pi
        xmc_dot_xgse = np.cos(gamma)*np.sin(tita)*np.cos(phi)-np.sin(gamma)*np.sin(phi)
        if xmc_dot_xgse < 0:
            print('Error en el cuadrante de gamma')
    return gamma


def rotacion(x_gse, y_gse, z_gse, tita_deg, phi_deg):

    tita= tita_deg*np.pi/180.
    phi = phi_deg*np.pi/180.

    tan_phi = np.tan(phi)
    gamma = calculo_gamma(phi, tita)

    R= np.zeros((3,3))
    R[0][0]=np.cos(gamma)*np.sin(tita)*np.cos(phi)-np.sin(gamma)*np.sin(phi)
    R[0][1]=np.cos(gamma)*np.sin(tita)*np.sin(phi)+np.sin(gamma)*np.cos(phi)
    R[0][2]=-np.cos(gamma)*np.cos(tita)
    R[1][0]=-np.sin(gamma)*np.sin(tita)*np.cos(phi)-np.cos(gamma)*np.sin(phi)
    R[1][1]=-np.sin(gamma)*np.sin(tita)*np.sin(phi)+np.cos(gamma)*np.cos(phi)
    R[1][2]=np.sin(gamma)*np.cos(tita)
    R[2][0]=np.cos(tita)*np.cos(phi)
    R[2][1]=np.cos(tita)*np.sin(phi)
    R[2][2]=np.sin(tita)

    x_mc = R[0][0]*x_gse+R[0][1]*y_gse+R[0][2]*z_gse
    y_mc = R[1][0]*x_gse+R[1][1]*y_gse+R[1][2]*z_gse
    z_mc = R[2][0]*x_gse+R[2][1]*y_gse+R[2][2]*z_gse

    return x_mc, y_mc, z_mc
