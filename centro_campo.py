# codigo moficado el dia 23-marzo 2020
# dado que este valor de CB daba el doble de lo que
# deberia dar. pues en el numerador falta dividir por
# tend - tstart

import numpy as np
import numpy.ma as ma


def centro_B(t, B):
    B0 = B_cero(t,B)
    integrando = [0] * len(B)
    tend = t[len(t)-1]
    tstart = t[0]
    tc = (tstart + tend) / 2
    for i in range(len(B)):
        if str(B[i])=='nan':
            t_norm[i]=np.nan
            integrando[i]=np.nan
        else:
#            integrando[i] = B[i]*t_norm[i]     linea de codigo vieja
            integrando[i] = ((t[i] - tc) / (tend - tstart)) * B[i]
    cleaned_int = [x for x in integrando if str(x) != 'nan']
    cleaned_t = [x for x in t if str(x) != 'nan']
    CB = np.trapz(cleaned_int, cleaned_t)
    return CB/B0

def B_cero(t,B):
    B_0 = np.trapz(B,t)
    return B_0
