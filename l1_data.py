import pyspedas
from pytplot import tplot, tplot_names, get_data, store_data
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta, datetime
import datetime as dt
from ftplib import FTP_TLS
import os
import matplotlib.dates as mdates
import download_data as dw
import matplotlib.transforms
import rotacion
import centro_campo
import thetaphi2lambdai as tr

#-----------------------------------------------------------------------
# Shock - ICME arrival time --------------------------------------------
#-----------------------------------------------------------------------
# Datos obtenidos de Piersanti Mirko 2020
#start_shock = datetime(2018, 8, 24, 5, 43, 0, 0)
#start_icme  = datetime(2018, 8, 25, 12, 15, 0, 0)
#end_icme    = datetime(2018, 8, 26, 10, 00, 0, 0)
start_icme1  = datetime(2018, 8, 24, 11, 20, 0, 0)
end_icme1    = datetime(2018, 8, 24, 22, 0, 0, 0)

#start_icme2  = datetime(2018, 8, 25, 12, 45, 0, 0)
start_icme2  = datetime(2018, 8, 25, 12, 50, 0, 0)
end_icme2    = datetime(2018, 8, 26, 7, 50, 0, 0)




#-----------------------------------------------------------------------
# Orientacion con MV ---------------------------------------------------
#-----------------------------------------------------------------------
theta_ICME1 = -10.6266
phi_ICME1 = 253.6616
R_ICME1 = 0.0423

theta_ICME2 = -44.6947
phi_ICME2 = 209.2911
R_ICME2 = 0.0721

l_ICME1, i_ICME1 = tr.titaphi2lambdai(theta_ICME1, phi_ICME1)
l_ICME2, i_ICME2 = tr.titaphi2lambdai(theta_ICME2, phi_ICME2)

#-----------------------------------------------------------------------
# Get data (ace) -------------------------------------------------------
#-----------------------------------------------------------------------
file = os.path.join(os.getcwd(), 'kp1808.wdc')
trange = ['2018-08-23', '2018-08-31']
ace_mfi_vars = pyspedas.ace.mfi(trange)
ace_swe_vars = pyspedas.ace.swe(trange)

# Extract Data ace
times_ace, B_ace = get_data('Magnitude')
B_ace = np.ma.masked_where(B_ace > 100., B_ace)
times_ace, Bc_ace = get_data('BGSEc')
Bx_ace = np.ma.masked_where(B_ace > 100., Bc_ace[:,0])
By_ace = np.ma.masked_where(B_ace > 100., Bc_ace[:,1])
Bz_ace = np.ma.masked_where(B_ace > 100., Bc_ace[:,2])
times_ace2, V_ace = get_data('Vp')
V_ace = np.ma.masked_where(V_ace == -999.9, V_ace)

start_icme2_ts = (start_icme2 - datetime(1970, 1, 1)).total_seconds()
end_icme2_ts = (end_icme2 - datetime(1970, 1, 1)).total_seconds()
#end_icme2.timestamp()
mask_icme2 = ((times_ace > start_icme2_ts) & (times_ace < end_icme2_ts))
V_nube2 = V_ace[mask_icme2]
Vmean_nube2 = np.ma.mean(V_nube2)
B_nube2 = B_ace[mask_icme2]
t_nube2 = times_ace[mask_icme2]

start_icme1_ts = (start_icme1 - datetime(1970, 1, 1)).total_seconds()
end_icme1_ts = (end_icme1 - datetime(1970, 1, 1)).total_seconds()
mask_icme1 = ((times_ace > start_icme1_ts) & (times_ace < end_icme1_ts))
V_nube1 = V_ace[mask_icme1]
Vmean_nube1 = np.ma.mean(V_nube1)
t_nube1 = times_ace[mask_icme1]
B_nube1 = B_ace[mask_icme1]

times_ace2, Np_ace =  get_data('Np')
Np_ace = np.ma.masked_where(Np_ace == -999.9, Np_ace)
times_ace2, Tp_ace = get_data('Tpr')
Tp_ace = np.ma.masked_where(Tp_ace == -1000., Tp_ace)

dates_ace=[datetime.utcfromtimestamp(ts) for ts in times_ace]


# Omni data
omni_vars = pyspedas.omni.data(trange)
times_omni, beta = get_data('Beta')
times_omni, B_omni = get_data('F')
times_omni, V_omni = get_data('flow_speed')
times_omni, time_shift = get_data('Timeshift')
times_omni, Vx_omni = get_data('Vx')
times_omni, Tp_omni = get_data('T')
times_omni, Np_omni = get_data('proton_density')
shift_time_omni = times_omni - time_shift
dates_omni=[datetime.utcfromtimestamp(ts) for ts in times_omni]
dates_omni_shift=[datetime.utcfromtimestamp(ts) for ts in shift_time_omni]
#wind_mfi_vars = pyspedas.wind.mfi(trange)
#times_wind, B_wind = get_data('BF1')
#wind_swe_vars = pyspedas.wind.swe(trange)
#times_wind, Bc_wind = get_data('B3GSE')
#dates_wind = [datetime.utcfromtimestamp(ts) for ts in times_wind]

#-----------------------------------------------------------------------
# Get data kp data -----------------------------------------------------
#-----------------------------------------------------------------------
# Create 3hour bin interval for kp plot
start = datetime.strptime(trange[0], '%Y-%m-%d')
end = datetime.strptime(trange[1], '%Y-%m-%d') + timedelta(days=1)
def datetime_range(start, end, delta):
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta

data_full = dw.kp(file)
datas = [ data_full[i][12:28] for i in range(len(data_full))]
kp_data = [float(d[idx:idx+2])/10 for d in datas for idx,val in enumerate(d) if idx%2 == 0]

kp=np.asarray(kp_data,dtype=np.float32)
start = datetime.strptime('2018-08-01', '%Y-%m-%d')
end = datetime.strptime('2018-08-31', '%Y-%m-%d') + timedelta(days=1)
date_plot = [dt for dt in datetime_range(start, end, {'days': 0, 'hours':3})]
date_plot = np.asarray(date_plot)
date_plot = date_plot[0:len(date_plot)]

#------------------------------------------------------------------------
# Get data dst data -----------------------------------------------------
#------------------------------------------------------------------------
fecha = trange[0][0:4]+trange[0][5:7]
tiempo_dst, dst_data = dw.dst(fecha)

#------------------------------------------------------------------------
# Plot data -------------------------------------------------------------
#------------------------------------------------------------------------
#Events limits
fig, axes= plt.subplots(7,1, sharex=True, figsize=(12,15))
colorline = 'red'
lw = 0.8
fs=10
fs_label = 12
fs_tick = 12
fs_legend = 12
start = datetime.strptime(trange[0], '%Y-%m-%d')
end = datetime.strptime(trange[1], '%Y-%m-%d')

CB_ICME2 = centro_campo.centro_B(t_nube2, B_nube2)
CB_ICME2 = round(CB_ICME2, 3)
CB_ICME1 = centro_campo.centro_B(t_nube1, B_nube1)
CB_ICME1 = round(CB_ICME1, 3)
axes[0].plot(dates_ace, B_ace, label='B_ace', color=colorline)
'''
axes[0].plot(dates_ace, Bx_ace, label='$B_{x,GSE}$', color = 'tab:blue', linewidth = lw)
axes[0].plot(dates_ace, By_ace, label='$B_{y,GSE}$', color = 'green',linewidth = lw)
axes[0].plot(dates_ace, Bz_ace, label='$B_{z,GSE}$', color = 'orange',linewidth = lw)
axes[0].set_xlim([start, end])
axes[0].legend(labelcolor='linecolor',handlelength=0, handletextpad=0,
        loc='upper left', bbox_to_anchor=(0.74, 1.05), frameon=False,
        labelspacing=0.2, fontsize=fs_legend, ncol=2)
axes[0].set_ylabel('IMF [nT]',fontsize=fs_label)
'''
axes[0].set_ylabel('B [nT]',fontsize=fs_label)
axes[0].hlines(0, start, end,linestyles = '--',color='gray', linewidth=0.5)
str_CB = "$C_{B,ICME1}$ = " + str(CB_ICME1) + "\n$C_{B,ICME2}$ = " + str(CB_ICME2)
axes[0].text(0.02, 0.98, str_CB ,horizontalalignment='left',
        verticalalignment='top',transform=axes[0].transAxes,
        color='black', fontsize=fs)
axes[0].text(0.215, -0.1, 'ICME 1' ,horizontalalignment='center',
        verticalalignment='bottom',transform=axes[0].transAxes,
        color='black', fontsize=fs)
axes[0].text(0.37, -0.1, 'ICME 2' ,horizontalalignment='center',
        verticalalignment='bottom',transform=axes[0].transAxes,
        color='black', fontsize=fs)


axes[1].plot(dates_ace, V_ace, color=colorline)
axes[1].set_ylabel('V [kms$^{-1}$]',fontsize=fs_label)
axes[1].axhspan(650, 500, facecolor='lightsalmon', alpha=0.2)
axes[1].axhspan(500, 400, facecolor='palegreen', alpha=0.2)
axes[1].text(0.02, 0.6, 'Fast',horizontalalignment='left',
        verticalalignment='bottom',transform=axes[1].transAxes,
        color='salmon', fontsize=fs)
axes[1].axhspan(400, 300, facecolor='#c7dbf0', alpha=0.5)

axes[1].text(0.02, 0.15, 'Slow',horizontalalignment='left',
        verticalalignment='bottom',transform=axes[1].transAxes,
        color='cornflowerblue', fontsize=fs)
axes[1].hlines(Vmean_nube2, start_icme2, end_icme2, color='black', linestyles='solid', linewidth=1)
axes[1].hlines(Vmean_nube1, start_icme1, end_icme1, color='black', linestyles='solid', linewidth=1)
str_V = '$V_{mean,ICME 1}=$'+str(int(Vmean_nube1))+'kms$^{-1}$' + '\n$V_{mean,ICME 2}=$'+str(int(Vmean_nube2))+'kms$^{-1}$'
axes[1].text(0.02, 0.98, str_V ,horizontalalignment='left',
        verticalalignment='top',transform=axes[1].transAxes,
        color='black', fontsize=fs)
axes[1].set_xlim([start, end])
axes[1].set_ylim([300, 650])

axes[2].plot(dates_omni_shift, beta, color=colorline)
axes[2].set_ylabel(r'$\beta$',fontsize=fs_label)
axes[2].set_yscale('log')
axes[2].set_ylim((0.1),(500))
axes[2].set_xlim([start, end])
axes[2].hlines(1, start, end,linestyles = '--',color='gray', linewidth=0.5)

axes[3].plot(dates_ace, Np_ace, color=colorline)
axes[3].set_ylabel('Np [cm$^{-3}$]',fontsize=fs_label)
axes[3].set_xlim([start, end])

axes[4].plot(dates_ace, Tp_ace, color=colorline)
axes[4].set_ylabel('Temp [K]',fontsize=fs_label)
axes[4].set_yscale('log')
axes[4].set_ylim((5*10**3),(10**6))
axes[4].set_xlim([start, end])


mask0 = kp<4.0
mask1 = ((kp >=4.0) & (kp < 5.0))
mask2 = ((kp >=5.0) & (kp < 6.0))
mask3 = ((kp >=6.0) & (kp<7.0))
mask4 = ((kp >=7.0) & (kp<8.0))
mask5 = ((kp >=8.0) & (kp<=9.0))

axes[5].hlines(5, start, end, color='#ffcb08', linestyles='solid', linewidth=0.5)
axes[5].text(0.02, 0.50, 'G1',horizontalalignment='left',verticalalignment='bottom',transform=axes[5].transAxes,color='#ffcb08', fontsize=fs)
axes[5].hlines(6, start, end, color='#faa61a', linestyles='solid', linewidth=0.5)
axes[5].text(0.02, 0.60, 'G2',horizontalalignment='left',verticalalignment='bottom',transform=axes[5].transAxes,color='#faa61a', fontsize=fs)
axes[5].hlines(7, start, end, color='#f36f21', linestyles='solid', linewidth=0.5)
axes[5].text(0.02, 0.70, 'G3',horizontalalignment='left',verticalalignment='bottom',transform=axes[5].transAxes,color='#f36f21', fontsize=fs)
axes[5].hlines(8, start, end, color='#c9252b', linestyles='solid', linewidth=0.5)
axes[5].text(0.02, 0.80, 'G4',horizontalalignment='left',verticalalignment='bottom',transform=axes[5].transAxes,color='#c9252b', fontsize=fs)
axes[5].text(0.02, 0.90, 'G5',horizontalalignment='left',verticalalignment='bottom',transform=axes[5].transAxes,color='#c9252b', fontsize=fs)
axes[5].hlines(9, start, end, color='#c9252b', linestyles='solid', linewidth=0.5)
axes[5].bar(date_plot[mask0], kp[mask0], align='edge', width=0.1, color='#c7dbf0')
axes[5].bar(date_plot[mask1], kp[mask1], align='edge', width=0.1, color='#97d33c')
axes[5].bar(date_plot[mask2], kp[mask2], align='edge', width=0.1, color = '#ffcb08')
axes[5].bar(date_plot[mask3], kp[mask3], align='edge', width=0.1, color = '#faa61a')
axes[5].bar(date_plot[mask4], kp[mask4], align='edge', width=0.1, color = '#f36f21')
axes[5].bar(date_plot[mask5], kp[mask5], align='edge', width=0.1, color = '#c9252b')
axes[5].set_xlim([start, end])
axes[5].set_ylim([0,10])
axes[5].set_ylabel('kp',fontsize=fs_label)

axes[6].plot(tiempo_dst, dst_data, color='black')
axes[6].hlines(0, start, end,linestyles = '--',color='gray', linewidth=0.5)
axes[6].set_xlim([start, end])
axes[6].set_ylim(-200,50)
axes[6].set_ylabel('Dst [nT]',fontsize=fs_label)
myFmt = mdates.DateFormatter('%d')
axes[6].xaxis.set_major_formatter(myFmt)
axes[6].text(0.02, 0.30,'Intense',
        verticalalignment='bottom', horizontalalignment='left',
        transform=axes[6].transAxes,
        color='#ef3f23', fontsize=fs)
axes[6].text(0.02, 0.50, 'Moderate',
        verticalalignment='bottom', horizontalalignment='left',
        transform=axes[6].transAxes,
        color='#f36f21', fontsize=fs)
axes[6].text(0.02,0.60, 'Small',
        verticalalignment='bottom', horizontalalignment='left',
        transform=axes[6].transAxes,
        color='#faa61a', fontsize=fs)
axes[6].hlines(-30, start, end, color='#ffcb08', linewidth=0.5)
axes[6].hlines(-50, start, end,color='#faa61a', linewidth=0.5)
axes[6].hlines(-100, start, end,color='#f36f21', linewidth=0.5)
axes[6].axhspan(50, -30, facecolor='#c7dbf0', alpha=0.5)
axes[6].axhspan(-30, -50, facecolor='#ffcb08', alpha=0.5)
axes[6].axhspan(-50, -100, facecolor='#faa61a', alpha=0.5)
axes[6].axhspan(-100, -200, facecolor='#f36f21', alpha=0.5)
axes[6].set_xlabel('Time [days]\n August/2018', fontsize=fs_label)

cuadro = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']
for i in range(7):
    axes[i].axvline(x=start_icme1, c="blue",linewidth=0.5)
    axes[i].axvline(x=end_icme1, c="blue",linewidth=0.5)
    axes[i].axvline(x=start_icme2, c="blue",linewidth=0.5)
    axes[i].axvline(x=end_icme2, c="blue",linewidth=0.5)
    axes[i].tick_params(axis='both', which='major', labelsize=fs_tick)
    axes[i].axvspan(start_icme1, end_icme1, facecolor='#c7dbf0', alpha=0.5)
    axes[i].axvspan(start_icme2, end_icme2, facecolor='#c7dbf0', alpha=0.5)
    axes[i].text(0.98, 0.98, cuadro[i] ,horizontalalignment='right',
            verticalalignment='top',transform=axes[i].transAxes,
            color='black', fontsize=fs_legend)

fig.align_labels()
#plt.plot(dates_wind, Bc_wind[:,0])
plt.savefig('interplanetary_conditions.png', bbox_inches='tight', dpi=300)
plt.savefig('interplanetary_conditions.pdf', bbox_inches='tight')
plt.close()


#------------------------------------------------------------------------
# Plot data in FR coordinates--------------------------------------------
#------------------------------------------------------------------------
#Events limits
fig2, axes= plt.subplots(3,1, sharex=True, figsize=(5,5))
colorline = 'blue'
lw = 0.8
fs=7
fs_label = 10
fs_tick = 8
fs_legend = 8
start = datetime.strptime(trange[0], '%Y-%m-%d') + timedelta(days=2)
end = datetime.strptime('2018-08-29', '%Y-%m-%d') - timedelta(days=2)

BxFR_2, ByFR_2, BzFR_2 = rotacion.rotacion(Bx_ace, By_ace, Bz_ace, theta_ICME2, phi_ICME2)
BxFR_nube2 = BxFR_2[mask_icme2]
ByFR_nube2 = ByFR_2[mask_icme2]
BzFR_nube2 = BzFR_2[mask_icme2]
#BxFR_nube2_mean = np.mean(BxFR_nube2)
#Bo2 = np.max(ByFR_nube2)
#p_gulisano = R*np.sqrt( np.abs(BxFR_nube2_mean/Bo2)/1.6)
#p = np.round(p_gulisano, 2)


#B = (Bx_ace**2 + By_ace**2 + Bz_ace**2)**(0.5)
#BFR = (BxFR**2 + ByFR**2 + BzFR**2)**(0.5)

'''
axes[0].plot(dates_ace, B_ace, label='B', color='black', linewidth = lw)
axes[0].plot(dates_ace, B, label='B', color='blue', linewidth = lw)
axes[0].plot(dates_ace, BFR, label='B', color='red', linewidth = lw)
axes[0].set_ylabel('B [nT]',fontsize=fs_label)
'''
axes[0].plot(dates_ace, BxFR_2, label='FR frame', color = 'red', linewidth = lw)
axes[0].plot(dates_ace, Bx_ace, label='GSE frame', color = 'black', linewidth = lw)
axes[0].set_ylim(-20, 20)
axes[0].set_ylabel('$B_{x}$ [nT]',fontsize=fs_label)
str_MV = '$\\theta_{MV}=$'+str(int(theta_ICME2))+ '$^\circ$, $\phi_{MV}=$' + str(int(phi_ICME2)) +'$^\circ$ \n $\lambda_{MV}=$' + str(int(l_ICME2)) + '$^\circ$, $i_{MV}=$' +str(int(i_ICME2)) +'$^\circ$'# \n $p$=' + str(p)
axes[0].text(0.02, 0.98, str_MV ,horizontalalignment='left',
        verticalalignment='top',transform=axes[0].transAxes,
        color='black', fontsize=fs)

axes[1].plot(dates_ace, ByFR_2, color = 'red',linewidth = lw)
axes[1].plot(dates_ace, By_ace, color = 'black',linewidth = lw)
axes[1].set_ylabel('$B_{y}$ [nT]',fontsize=fs_label)
axes[1].set_ylim(-20, 20)
axes[2].plot(dates_ace, BzFR_2, color = 'red',linewidth = lw)
axes[2].plot(dates_ace, Bz_ace, color = 'black',linewidth = lw)
axes[2].set_ylabel('$B_{z}$ [nT]',fontsize=fs_label)
axes[2].set_xlim([start, end])
axes[2].set_ylim(-20, 20)
axes[0].legend(loc='upper right', fontsize=fs_legend)


for i in range(3):
    axes[i].axhline(y=0, color='black')
    axes[i].axvline(x=start_icme2, c="blue",linewidth=0.5)
    axes[i].axvline(x=end_icme2, c="blue",linewidth=0.5)
    axes[i].tick_params(axis='both', which='major', labelsize=fs_tick)


for i in range(3):
    axes[i].axvspan(start_icme2, end_icme2, facecolor='#c7dbf0', alpha=0.5)

myFmt = mdates.DateFormatter('%d %H')
axes[2].xaxis.set_major_formatter(myFmt)
axes[2].set_xlabel('Time [days hour] \n August/2018', fontsize=fs_label)
plt.savefig('ICME2.png', bbox_inches='tight', dpi=300)
plt.savefig('ICME2.pdf', bbox_inches='tight')
plt.close()


# --------------------------------------------------------------------------
# Plot ICME1 --------------------------------------------------------------
# --------------------------------------------------------------------------
#Events limits
fig4, axes= plt.subplots(3,1, sharex=True, figsize=(5,5))
colorline = 'blue'
lw = 0.8
fs=7
fs_label = 10
fs_tick = 8
fs_legend = 8

start = start_icme1 - timedelta(days=1)
end = end_icme1 + timedelta(days=1)

BxFR_1, ByFR_1, BzFR_1 = rotacion.rotacion(Bx_ace, By_ace, Bz_ace, theta_ICME1, phi_ICME1)
BxFR_nube1 = BxFR_1[mask_icme1]
ByFR_nube1 = ByFR_1[mask_icme1]
BzFR_nube1 = BzFR_1[mask_icme1]
#BxFR_nube_mean = np.mean(BxFR_nube1)
#Bo = np.max(ByFR_nube)
#p_gulisano = R_ICME1*np.sqrt( np.abs(BxFR_nube_mean/Bo)/1.6)
#p = np.round(p_gulisano, 2)


#B = (Bx_ace**2 + By_ace**2 + Bz_ace**2)**(0.5)
#BFR = (BxFR**2 + ByFR**2 + BzFR**2)**(0.5)

'''
axes[0].plot(dates_ace, B_ace, label='B', color='black', linewidth = lw)
axes[0].plot(dates_ace, B, label='B', color='blue', linewidth = lw)
axes[0].plot(dates_ace, BFR, label='B', color='red', linewidth = lw)
axes[0].set_ylabel('B [nT]',fontsize=fs_label)
'''
axes[0].plot(dates_ace, BxFR_1, label='FR frame', color = 'red', linewidth = lw)
axes[0].plot(dates_ace, Bx_ace, label='GSE frame', color = 'black', linewidth = lw)
axes[0].set_ylim(-10, 10)
axes[0].set_ylabel('$B_{x}$ [nT]',fontsize=fs_label)
str_MV = '$\\theta_{MV}=$'+str(int(theta_ICME1))+ '$^\circ$, $\phi_{MV}=$' + str(int(phi_ICME1)) +'$^\circ$ \n $\lambda_{MV}=$' + str(int(l_ICME1)) + '$^\circ$, $i_{MV}=$' +str(int(i_ICME1)) +'$^\circ$'# \n $p$=' + str(p)
axes[0].text(0.02, 0.98, str_MV ,horizontalalignment='left',
        verticalalignment='top',transform=axes[0].transAxes,
        color='black', fontsize=fs)

axes[1].plot(dates_ace, ByFR_1, color = 'red',linewidth = lw)
axes[1].plot(dates_ace, By_ace, color = 'black',linewidth = lw)
axes[1].set_ylabel('$B_{y}$ [nT]',fontsize=fs_label)
axes[1].set_ylim(-10, 10)

axes[2].plot(dates_ace, BzFR_1, color = 'red',linewidth = lw)
axes[2].plot(dates_ace, Bz_ace, color = 'black',linewidth = lw)
axes[2].set_ylabel('$B_{z}$ [nT]',fontsize=fs_label)
end = datetime(2018, 8, 26, 0, 0, 0, 0)
axes[2].set_xlim([start, end])
axes[2].set_ylim(-10, 10)
axes[0].legend(loc='upper right', fontsize=fs_legend)


for i in range(3):
    axes[i].axhline(y=0, color='black')
    axes[i].axvline(x=start_icme1, c="blue",linewidth=0.5)
    axes[i].axvline(x=end_icme1, c="blue",linewidth=0.5)
    axes[i].tick_params(axis='both', which='major', labelsize=fs_tick)


for i in range(3):
    axes[i].axvspan(start_icme1, end_icme1, facecolor='#c7dbf0', alpha=0.5)

myFmt = mdates.DateFormatter('%d %H')
axes[2].xaxis.set_major_formatter(myFmt)
axes[2].set_xlabel('Time [days hour] \n August/2018', fontsize=fs_label)
plt.savefig('ICME1', bbox_inches='tight', dpi=300)
plt.savefig('ICME1.pdf', bbox_inches='tight')
plt.close()


#------------------------------------------------------------------------
# Plot hodograf ---------------------------------------------------------
#------------------------------------------------------------------------
fig3, axes= plt.subplots(1,2, figsize=(12,5))
colorline = 'blue'
lw = 0.8
fs=7
fs_label = 14
fs_tick = 12
fs_legend = 8

axes[0].scatter(ByFR_nube1,BzFR_nube1,c=t_nube1, cmap='plasma')
axes[0].set_xlabel('$B_{y,FR}$ [nT]',fontsize=fs_label)
axes[0].set_ylabel('$B_{z,FR}$ [nT]',fontsize=fs_label)
axes[0].annotate('start', (ByFR_nube1[0], BzFR_nube1[0]))
axes[0].annotate('end', (ByFR_nube1[-1], BzFR_nube1[-1]))
axes[0].set_title('ICME 1')
axes[0].set_xlim([-8,8])
axes[0].set_ylim([-8,8])

axes[1].scatter(ByFR_nube2,BzFR_nube2,c=t_nube2, cmap='plasma')
axes[1].set_xlabel('$B_{y,FR}$ [nT]',fontsize=fs_label)
axes[1].set_ylabel('$B_{z,FR}$ [nT]',fontsize=fs_label)
axes[1].annotate('start', (ByFR_nube2[0], BzFR_nube2[0]))
axes[1].annotate('end', (ByFR_nube2[-1], BzFR_nube2[-1]))
axes[1].set_title('ICME 2')
axes[1].set_xlim([-20,20])
axes[1].set_ylim([-20,20])
plt.savefig('hodografa.png', bbox_inches='tight', dpi=300)
plt.savefig('hodrografa.pdf', bbox_inches='tight')
plt.close()
