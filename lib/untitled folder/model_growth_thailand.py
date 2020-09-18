"""
Fish Growth model and parameters
Author: Abderrazak Chahid  |  abderrazak-chahid.com | abderrazak.chahid@gmail.com

"""
import numpy as np
import pandas as pd
import time

#%% List of  Input parameter of the ODE
# Feeding level
Fmin=0.01;
Fmax=1;
# Feed_Q=rand_vec(N,0, 1);                             #  feeding qunatity

# Water temperature
Tmin= 27.64;
Tmax=31.76;
Topt= 29.7;
# Temp=rand_vec(N,Tmin,Tmax);#Topt*ones(1,N);#rand_vec(N,Tmin,Tmax);#
# Temp=rand_vec(N,Topt-15,Topt+15);#Topt*ones(1,N);#rand_vec(N,Tmin,Tmax);#

# Dissolved oxygen (DO)
DOcri= 1;
DOmin= 0.3;

# DO=rand_vec(N,DOcri+1, DOcri+5); # optimal DO>DOcri
# DO=rand_vec(N,DOmin-0.1, DOcri+0.1);

#Unionized ammonia (UIA)
UIAmax=1.40;
UIAcri=0.06;
# UIA=rand_vec(N,UIAcri-5, UIAcri-0.1);# optimal UIA<UIAcri
# UIA=rand_vec(N,UIAcri-1, UIAmax+0.1);#

# Other
m=0.67;         # the exponent of body weight for net anabolism
n=0.81;         # the exponent of body weight for fasting catabolism
a=0.53;         # accounts for further losses of metabolizable energy via heat increment and urinary excretion
b=0.62;         # refers to the proportion of the gross energy or food intake that is available as metabolizable energy
kmin=0.00133;   # the coefficient of fasting catabolism
j=0.0132;

ru=0.8;
tau=0.9

# relative feeding level ( f )
s= 21.38;     # the proportionality coefficient of PNPP to natural food (dimensionless) was estimated
X= 21.38;     # the proportionality coefficient of PNPP to natural food (dimensionless) was estimated
Dn=-1;        # the DIN (mg/L*N)
lamba=-1;     #  is the efficiency of carbon fixation (dimensionless),
A= 85;        # is the alkalinity (mg/L*CaCO3),
Hplus=8.1;    # is the hydrogen ion concentration   (mol/L),

###################################################################################################################
def create_tank_env(N):
    for i in range(N):
        Temp = np.random.uniform(low=Tmin, high=Tmax, size=(N))
        DO = np.random.uniform(low=DOmin, high=1.2*DOcri, size=(N))
        UIA = np.random.uniform(low=0.8*UIAcri, high=UIAmax, size=(N))
    return Temp, DO, UIA

def load_growth_profile(filename):
    data = pd.read_csv(filename)
    t_data = data['t'].to_numpy()
    xf_data0 = data['xf'].to_numpy()
    dt = np.max( np.diff(t_data) )

    if filename.find('week'):
        dt=dt*7
    return t_data, xf_data0, dt

def zero_order_hold(v, L):
    o=0*v;
    N=len(v)
    for i in range(N):
        if i<N-L:
            if i%L==0:
                o[i]=value=np.mean(v[i:i+L])
                # print('new=',value)
            else:
                o[i]=value
                # print('hold=',value)
        else:

            o[i]=v[-1]
    return o

###################################################################################################################
filename="./data/Experimetal_data_Tilapia_Thailand_processed_weeks.csv"
t_data, xf_data0, dt= load_growth_profile(filename)
x0=xf_data0[0]; xf=xf_data0[-1];                # intial/final fish weight
# xf_data0=xf+0*xf_data0
N=len(xf_data0)
