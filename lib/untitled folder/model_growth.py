"""
Fish Growth model and parameters
Author: Abderrazak Chahid  |  abderrazak-chahid.com | abderrazak.chahid@gmail.com

"""
import numpy as np
import pandas as pd
import time
import math
### List of  Input parameter of the ODE
# time parameters
N=100
dt=1                                # sampling time

# Feeding level
Fmin=0
Fmax=1
# Feed_Q=rand_vec(N,0, 1)                             #  feeding qunatity

# Water temperature
Tmin= 26
Tmax=40
Topt= 33
# Temp=rand_vec(N,Tmin,Tmax)#Topt*ones(1,N)#rand_vec(N,Tmin,Tmax)#
# Temp=rand_vec(N,Topt-15,Topt+15)#Topt*ones(1,N)#rand_vec(N,Tmin,Tmax)#

# Dissolved oxygen (DO)
DOcri= 1
DOmin= 0.3

# DO=rand_vec(N,DOcri+1, DOcri+5) # optimal DO>DOcri
# DO=rand_vec(N,DOmin-0.1, DOcri+0.1)

#Unionized ammonia (UIA)
UIAmax=1.40
UIAcri=0.06
# UIA=rand_vec(N,UIAcri-5, UIAcri-0.1)# optimal UIA<UIAcri
# UIA=rand_vec(N,UIAcri-1, UIAmax+0.1)#

# Other
m=0.67         # the exponent of body weight for net anabolism
n=0.81         # the exponent of body weight for fasting catabolism
a=0.53         # accounts for further losses of metabolizable energy via heat increment and urinary excretion
b=0.62         # refers to the proportion of the gross energy or food intake that is available as metabolizable energy
kmin=0.00133   # the coefficient of fasting catabolism
j=0.0132

ru=0.8
tau=0.9

# relative feeding level ( f )
s= 21.38     # the proportionality coefficient of PNPP to natural food (dimensionless) was estimated
X= 21.38     # the proportionality coefficient of PNPP to natural food (dimensionless) was estimated
Dn=-1        # the DIN (mg/L*N)
lamba=-1     #  is the efficiency of carbon fixation (dimensionless),
A= 85        # is the alkalinity (mg/L*CaCO3),
Hplus=8.1    # is the hydrogen ion concentration   (mol/L),

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
    return t_data, xf_data0, dt

def zero_order_hold(v, L):
    o=0*v
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


def day_length(Day,Latitude):
    Axis=23.439*np.pi/180
    j=np.pi/182.625            #constant (radians)
    m=1-np.tan(Latitude*np.pi/180)*np.tan(Axis*np.cos(j*Day))

#saturate value for artic
    if m>2 :
        m=2
    if m<0 :
        m=0
#fraction of the day the sun is up
    b=math.acos(1-m)/np.pi

# Hours of sunlight
    hours=b*24

    return hours

def Fish_Growth_Model(x, f, T, DO, UIA):
    # global   m n a b s R Tmin Tmax Topt  kmin DOcri DOmin UIAmax UIAcri j noise

    ## generate noise
    Max=1
    Min=0
    ## save generated noise
    # consider the optimal value
    #  tau=(1-Gstar)*tau + Gstar*1

    ## Compute (tau): the effects of temperature on food consumption and therefore anabolism using the function
    tau=1
    if T>Topt:
        tau= np.exp( -4.6*((Topt-T)/(Topt-Tmin))**4 )

    else:
        tau= np.exp( -4.6*((T-Topt)/(Tmax-Topt))**4 )

    ## coefficient of catabolism (k) increases  with temperature.
    # k= kmin#   # (g**(1-n)/ day)

    k= kmin*np.exp(j*(T-Tmin) )   # (g**(1-n)/ day)
    # k=(1-noise)*k + noise*rand_vec(1,0,kmin)

    # consider the optimal value
    # k=(1-Gstar)*k + Gstar*kmin

    ## Photoperiod factor (ru) 0< ru < 2: many cultured fish species including tilapias tended  to feed only during daylight hours.
    from datetime import datetime
    Day = datetime.now().timetuple().tm_yday
    Lat_jeddah=21.4858
    Photoperiod_jeddah=day_length(Day,Lat_jeddah)

    ru=Photoperiod_jeddah/12

    # consider the optimal value
    # ru=(1-Gstar)*ru + Gstar*1

    ## the coefficient of food consumption
    h=0.8

    # consider the optimal value
    # h=(1-Gstar)*h + Gstar*1

    ## food consumption  affected when DO
    sigma=1

    if DO>DOcri:
         sigma=1

    elif DO>DOmin:
         sigma= (DO-DOmin)/(DOcri-DOmin)

    else:
        sigma=0


    # consider the optimal value
    # sigma=(1-Gstar)*sigma + Gstar*1
    ## food consumption  affected when UIA
    v=1

    if UIA<UIAcri:
         v=1
    elif UIA<UIAmax:
         v= (UIAmax-UIA)/(UIAmax-UIAcri)

    else:
        v=0

    ## Equation 1 : relative feeding level f(r)
       # r is the actual daily ration (g/day),
       # R is the maximal daily ration (g/day),
    # f=F#r/R
    #
    # ## Equation 2 : relative feeding level f(T, Ph, DIP, DIN,...)
    #
    # ##  P is concentration of natural food ( g/m3)
    #    # lamba is the efficiency of carbon fixation (dimensionless),
    #    # A is the alkalinity (mg l1 CaCO3),
    #    # Hplus is the hydrogen ion concentration   (mol/L),
    #    # k1the first dissociation constant for carbonate:bicarbonate system
    #    # k2 is the the second dissociation constant for carbonate:bicarbonate system
    #    # T is the water temperature (°C).
    #    # The constants of 12 and 50  are gram equivalent weights of C and CaCO3
    #
    # k1=(T/15 + 2.6)*10**-7
    # k2=(T/10 + 2.2)*10**-11
    #
    # Pc=12*lamba*(A/50)*( (Hplus**2)/k1 +  Hplus + k2)/( Hplus + 2*k2 )
    #
    # ## Pn is the PNPP derived from total dissolved inorganic nitrogen DIN [ g/(m3*day*C) ]
    #     # Dn is the DIN (mg/L*N),
    # Pn=40*Dn/7
    #
    # ## Pp is the PNPP derived from total dissolved inorganic phosphorus DIP (mg/L*N),
    #     # Dp is the  DIP (mg/L*N),
    # Pp=40*Dp
    #
    # ## relative feeding level ( f ) with potential net primary productivity (PNPP) [ g/(m3*day*C) ]
    # P= min([Pc, Pn, Pp])
    #
    # ## relative feeding level ( f )  Equation 1
    #     # s is the proportionality coefficient of PNPP to natural food (dimensionless) was estimated
    # f=1-np.exp(-s*(P/B))



    ## fish growth rate

    ## DOE system
    C1=b*(1-a)*ru*tau*f
    # xdot =0.7658*f*x**m - 0.0013*x**n
    xdot =C1*x**m - k*x**n

    print('\n\n--> Catabolism factor', C1)
    print('\n\n--> Metabolism factor', k)

    input('Model parameters:')
    return xdot

###################################################################################################################
# create the environment
Temp, DO, UIA = create_tank_env(N)
x0=13; xf=100                      # intial/final fish weight
xf_data= xf*np.ones(Temp.shape)             # constance profile
t_data=np.arange(start=0, stop=N, step=dt)
