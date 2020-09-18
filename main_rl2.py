"""
Reinforcement Learning using table lookup Q-learning method.
The RL state is a quatization of the weight state (x)
Author: Abderrazak Chahid  |  abderrazak-chahid.com | abderrazak.chahid@gmail.com

This script is modified based on the original source from: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time
import math
import pickle
import sys
import matplotlib.pyplot as plt;
import seaborn as sns; sns.set()
from lib.model_growth_thailand import * ;
# from lib.model_growth import * ; reward=1

import os
import scipy.io as sio
clear = lambda: os.system('clear')
import itertools

np.random.seed(2)  # reproducible

###################################################################################################################
# experiemnt parameters
RESET_RL=1                                    # 0) load trained q_table, 1) train RL
# eps_=1/4000                                 # decaying  greedy police : EPSILON = EPSILON*exp()
# ZH_=0                                       # apply zero-hlod filer to the tracking data
L_ZH=10                                       # zero_order_hold length
# reward_=1                                   #  choose on of the reward as defind in "get_env_feedback" routine
R_max= 0.03                                   # maximal feeding ration

# create the environment
Temp, DO, UIA = create_tank_env(N)

for alpha in [ 0.5]:                         # learning rate  ~0.05
    for gamma in [ 0.5]:                      # discount factor ~0.5
        for reward_ in [1]:#range(5,7):       # reward: 0) exponential   1) final target
            for ZH_ in range(0,1):
                # for eps_ in [1/400]:                  # greedy police~1/explotion episodes
                ###################################################################################################################
                # Q-learning parameters
                N_ACTIONS2=[7,7]                             # ~[3,3] the number of actions [F,T]
                WEIGHT_RESOLUTION=10#10                        # ~5 Discretiztion grid of the states (weight)
                TIME_RESOLUTION=10#3                           # ~2 Discretiztion grid of the time (t)
                EPSILON = 1#0.9                               # ~0.9 greedy police: 0) totally random    1) follows the policy
                # eps_    = 1/N_STATES                        # ~<1/N_STATES the simulated annealing  factor
                ALPHA = alpha                               # ~0.05 learning rate
                GAMMA = gamma                               # ~0.5 discount factor
                EPISODES_EVAL=500                           # ~500 interval to evaluate the learned policy
                FRESH_TIME = 0.0                            # ~0.0001 fresh time for one move
                POLICY_ERROR= 1.2                             # ~3 means we stop if the policy did not change after POLICY_ERROR*N_STATES iteration

                #################################################################################################################
                W_STATES = math.floor((xf-x0)/WEIGHT_RESOLUTION)                             # the number of culture  days++>  dimension of world

                TIME_ZONES=math.floor((t_data[-1]-t_data[0])/TIME_RESOLUTION)  ;
                N_STATES = W_STATES*TIME_ZONES
                eps_    = 1/400#N_STATES                        # ~<1/N_STATES the simulated annealing  factor
                MAX_FEED_RATIO=Fmax                          # The maximun tolerated feed ratio to the fish weight
                MIN_FEED_RATIO=Fmin
                FEEDING=[i for i in np.linspace(MIN_FEED_RATIO, MAX_FEED_RATIO, num=N_ACTIONS2[0])]   ## 'R': feed with amount equal to k*W where W is the fish weight,
                # HEATING=[i for i in np.linspace(Tmin, Tmax, num=N_ACTIONS2[1])]   ## 'R': feed with amount equal to k*W where W is the fish weight,
                HEATING=[i for i in np.linspace(Tmin+2, Tmax-1, num=N_ACTIONS2[1])]   ## 'R': feed with amount equal to k*W where W is the fish weight,
                ACTIONS, ACTIONS_value= get_actions2(FEEDING, HEATING)
                #
                # ACTIONS = ['R='+str(i)+'*W' for i in FEEDING]
                MAX_EPISODES = -1#1000*10*N_STATES*len(ACTIONS)     # maximum episodes
                # print(FEEDING); print(ACTIONS)
                #################################################################################################################

                IMPROVE_POLICY=[]
                TRACKING_MSE=[]
                TRACKING_MSE_eps=[]

                print('FEEDING=\n',FEEDING);print('HEATING=\n',HEATING);
                input('flag')


                if ZH_==1:
                    xf_data=zero_order_hold(xf_data0, L_ZH)

                else:
                    xf_data=xf_data0

                def moving_average(x_, L, step):
                    # L=100
                    # step=50
                    mean_k=[]; std_k=[]
                    k=0;
                    while k<len(x_)-L:
                        mean_k.append(np.mean(x_[k:k+L]))
                        std_k.append(np.std(x_[k:k+L]))

                        k+=step
                    mean_=np.array(mean_k)
                    std_=np.array(std_k)
                    t = np.linspace(1,len(x_), len(mean_))

                    return t, mean_, std_

                def plot_and_save_results(time, food, fish_weight, Temp, IMPROVE_POLICY, filename, suff):
                    from matplotlib.ticker import ScalarFormatter,AutoMinorLocator

                    print('\n-->Plot the fish growth responding to the RL-optimal feeding  \n')

                    total_food=dt*np.sum(R_max*(np.multiply(food,fish_weight)))
                    # total_energy=dt*np.sum(R_max*(np.multiply(food,Temp)))

                    fig = plt.figure()
                    ax0 = fig.add_subplot(211)
                    ax0.plot(t_data, xf_data0, '-.b',label='Experimental reference [ $W_{final}$= '+str(format(xf_data0[-1], '.1f'))+'g]')
                    if ZH_==1:
                        ax0.plot(t_data, xf_data, '-.k',label=' Tracking reference [ $W_{final}$= '+str(format(xf_data[-1], '.1f'))+'g]')
                    ax0.plot(time, fish_weight, 'r',label='Q-Learning    [ $W_{final}$= '+str(format(fish_weight[-1], '.1f'))+'g]')
                    ax0.set_title('Fish growth')
                    ax0.set_ylabel('Mean fish weight (g)')
                    ax0.grid()
                    ax0.legend(loc="upper left")
                    plt.tight_layout()


                    ax = fig.add_subplot(212)

                    lns1 = ax.plot(time, R_max*100*food, '-r', label = 'Feeding [Total food = '+str(format(total_food, '.1f'))+'g]')
                    ax2 = ax.twinx()
                    lns3 = ax2.plot(time, Temp, '-g', label = 'Water temperature')# [Total Energy = '+str(format(total_energy, '.1f'))+'g]')



                    ax.grid()
                    if dt==7:
                        ax.set_xlabel("Culture period (weeks)")
                    else:
                        ax.set_xlabel("Culture period (days)")
                    ax.set_ylabel('Percent of body weight (%)\nper day(BWD) ')
                    ax.set_title('Controlled feeding and temperature')



                    ax2.set_ylabel(r"Temperature ($^\circ$C)")
                    # ax2.set_ylim(0, 35)
                    # ax.set_ylim(-20,100)
                    ax2.grid()

                    # added these three lines
                    lns = lns1+lns3
                    labs = [l.get_label() for l in lns]
                    ax2.legend( lns, labs, loc=0)#, frameon=False)
                    ax.yaxis.set_major_formatter(ScalarFormatter())
                    plt.tight_layout()

                    fig.savefig(filename+'growth_'+suff+'.png', format='png', dpi=1200)

                    # plot the policy
                    plot_policy(IMPROVE_POLICY, filename, suff)

                    # save the results and parameters
                    save_results(t_data, xf_data0, xf_data, time, food, fish_weight, q_table, x0, xf, Temp, IMPROVE_POLICY, TRACKING_MSE_eps, TRACKING_MSE, EPISODES_EVAL, filename, suff)

                def plot_policy(IMPROVE_POLICY, filename, suff):


                    #%% policy iteration error
                    err=np.array(IMPROVE_POLICY)
                    L_frame, step= 100, 50
                    t, mean_, std_= moving_average(err, L_frame, step)

                    plt.figure()
                    plt.plot(t, mean_, 'r-')
                    plt.fill_between(t, mean_ - std_, mean_ + std_, color='r', alpha=0.2)
                    plt.title('Policy Improvement')
                    plt.ylabel('Policy  error')
                    plt.xlabel('Episodes')
                    # plt.legend()
                    plt.tight_layout()
                    plt.savefig(filename+'policy_iter_'+suff+'.png', format='png', dpi=1200)

                    plt.figure()
                    plt.scatter(TRACKING_MSE_eps, TRACKING_MSE)
                    plt.title('Policy evaluation [Exploitation -'+str(EPISODES_EVAL)+']')
                    plt.xlabel('Episodes')
                    plt.ylabel('Relative error(%)')
                    # plt.legend()
                    plt.tight_layout()
                    plt.savefig(filename+'policy_eval_'+suff+'.png', format='png', dpi=1200)

                def save_results(t_data, xf_data0, xf_data, time, food, fish_weight, q_table, x0, xf, Temp, IMPROVE_POLICY, TRACKING_MSE_eps, TRACKING_MSE, EPISODES_EVAL, filename, suff):
                    ##################   save the obtained results     ########################
                    print('\n-->Save the obtained feeding performance  \n')
                    #Pickle
                    with open(filename+'var__'+suff+'.pickle', 'wb') as f:
                        pickle.dump([t_data, xf_data0, xf_data, time, food, fish_weight, q_table, x0, xf, Temp, IMPROVE_POLICY, TRACKING_MSE_eps, TRACKING_MSE, EPISODES_EVAL], f)
                    # ## Load results
                    # #filename='./results2/RL2_var_'+suff+'.pickle'
                    # with open(filename, 'rb') as f:
                        # t_data, xf_data0, xf_data, time, food, fish_weight, q_table, x0, xf, Temp, IMPROVE_POLICY, TRACKING_MSE_eps, TRACKING_MSE, EPISODES_EVAL= pickle.load(f)

                    #%% save states in  csv file
                    d = {'t':time,'f':food,'T':Temp,'w':fish_weight,'t_data':t_data,'xf_data0':xf_data0,'xf_data':xf_data,
                        'q_table':q_table,'x0':x0,'xf':xf,'reward_':reward_,'ZH_':ZH_,'eps_':eps_,
                        'TRACKING_MSE':TRACKING_MSE,'TRACKING_MSE_eps':TRACKING_MSE_eps, }
                    # results = pd.DataFrame(d); results.to_csv (filename+'.csv', index = False, header=True)
                    # sio.savemat(filename+'mat__'+suff +'.mat', d )

                    #%% Save the obtained policy and Q-Table
                    print('\n-->Save the obtained policy and Q-Table \n');
                    # Nmse=np.min([len(fish_weight),len(fish_weight)])
                    # RMSE=np.square(np.subtract(fish_weight[:Nmse], xf_data0[:Nmse])).mean()
                    q_table.to_csv (filename+'_q_table_'+suff+'.csv', index = True, header=True)
                    # print('\r\nQ-table:\n')
                    # print(q_table)

                    q_policy=get_policy(q_table)
                    q_policy.to_csv (filename+'_q_policy_'+suff+'.csv', index = False, header=True)

                def discritize_states(v, v_cont, v_discrt):
                    # print('v_cont=',v_cont)
                    # print('v=',v)
                    # print('v_discrt=',v_discrt)
                    idx=0
                    for i in v_discrt:
                        if v_cont[i] <= v:
                            idx=i
                    # print('v=',v)
                    # print(idx)

                    return v_discrt[idx]

                def get_aquarum_state(W, W_cont, W_discrt, T, T_cont, T_discrt):

                    s1=discritize_states(W, W_cont, W_discrt)
                    n1=len(W_discrt)

                    s2=discritize_states(T, T_cont, T_discrt)
                    n2=len(T_discrt)

                    st=n1*s2+s1

                    return st

                def build_q_table(n_states, actions):
                    global IMPROVE_POLICY
                    filename='./q_table_weights2.pickle'
                    if os.path.isfile(filename):
                        with open(filename, 'rb') as f:
                            q_table0, episode0, N_STATES0, ACTIONS0, IMPROVE_POLICY= pickle.load(f)

                        if  n_states==N_STATES and actions==ACTIONS0:
                            table=q_table0
                            return table
                    table = pd.DataFrame(
                        np.zeros((n_states, len(actions))),     # q_table initial values
                        columns=actions,    # actions's name
                    )

                    table[actions[-1]][:]=1#[n_states-1]=10 # feed the minimum at the terminal state
                    # table[actions[0]][n_states-1]=10 # feed the minimum at the terminal state
                    # print(table)    # show table
                    return table

                def choose_action(state, q_table, epsilon):
                    # This is how to choose an action
                    state_actions = q_table.iloc[state, :]
                    if (np.random.uniform() > epsilon) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
                        action_name = np.random.choice(ACTIONS)
                        # print('\n\nGreedy')
                    else:   # act greedy
                        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
                        # action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas

                    return action_name

                def get_env_feedback(S, A, day):
                    global x, xf, xf_data
                    ## Get the feeding factor
                    # f=FEEDING[ACTIONS.index(A)]

                    # print('A=',A)
                    # print('ACTIONS_value=',ACTIONS_value)
                    f=ACTIONS_value[ACTIONS.index(A)][0]
                    T=ACTIONS_value[ACTIONS.index(A)][1]

                    sigma=2
                    # print('reward_=', reward_, 'x=', x, 'xf=', xf)
                    if reward_==0:
                        #R= -(  ((x-xf))**2 + 0.0*(f/Fmax)**2)  # policy1
                        R=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-xf)**2)/(sigma**2))

                    elif reward_==1:
                        R= -(((x-xf)/xf)**2 + 0.1*(f/Fmax)**2)




                    ## Compute the fish growth form the model
                    W_, x = growth_model(x, f, T, day) # get the fish size state
                    D_=discritize_states(day, t_float, t_int)

                    if D_ == TIME_ZONES-1 and W_ == W_STATES-1:   # terminate
                        S_ = 'terminal'
                        # R=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-xf)**2)/(sigma**2))

                    else:
                        S_= W_ + W_STATES*D_      # map the pair(W,t) states
                        # print('day=',day)
                        # print('W_=',W_, '/ ', W_STATES)
                        # print('S_=',S_)
                    # print('Updated x=',x, ' R=',R, ' S=', S, ' f=',f, ' T=', T, ' D=', D_, ' W=',W_)


                    return S_, R

                def growth_model(x, f, T, tk):

                    if tk>=len(xf_data):
                        tk=len(xf_data)-1

                    DO_k   = DO[tk]
                    UIA_k  = UIA[tk]
                    # compare with MPC: T=Topt, DO>Dmin  UIA<UIAmin
                    dx = Fish_Growth_Model(x, f, T, DO_k, UIA_k);
                    x= x+dx*dt
                    S_new=discritize_states(x, x_float, x_int)
                    # S_new=get_aquarum_state(x, x_float, x_int, Temp_k, T_cont, T_discrt)

                    return S_new, x

                def get_policy(q_table):

                    q_policy=  (q_table == q_table.max(axis=1)[:,None]).astype(int)
                    print('\r\nPolicy-table: reward=', reward_, ', Greedy-decay=', eps_, ', ALPHA=', ALPHA, ', GAMMA=', GAMMA)
                    print(q_table,'\n')
                    print(q_policy,'\n')
                    return q_policy

                def update_env(S, episode, step_counter):
                    # This is how environment be updated
                    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
                    if S == 'terminal':
                        interaction = 'Episode %s/%d: total days = %s \nFinal fish weight = %.3f / %s' % (episode+1, MAX_EPISODES, step_counter, x, xf)
                        print('\r{}'.format(interaction), end='')
                        # time.sleep(2)
                        print('\r                                ', end='')
                    else:

                        if N_STATES<10: #display the leaninng grid world for small states
                            # print(S)
                            env_list[S] = 'o'
                            interaction = ''.join(env_list)
                            print('\r{}'.format(interaction), end='')
                            time.sleep(FRESH_TIME)

                def rl(N_STATES, ACTIONS):
                    global x, xf, xf_data, t_data, IMPROVE_POLICY, TRACKING_MSE, TRACKING_MSE_eps
                    # #filename='./results2/RL2_var_'+suff+'.pickle'
                    q_table = build_q_table(N_STATES, ACTIONS)
                    # store the initial policy
                    q_policy=get_policy(q_table)

                    eps0=len(IMPROVE_POLICY)

                    episode=eps0
                    run_learning=True
                    cnt_policy=eps0
                    mse_tracking, err_policy_old=-1, 10*N_STATES
                    while run_learning:#in range(MAX_EPISODES):
                        step_counter = 0
                        S = 0
                        day=0
                        x=x0
                        xf=xf_data[step_counter]
                        is_terminated = False
                        # animation
                        update_env(S, episode, step_counter)

                        episode+=1

                        while not is_terminated:

                            epsilon=(1-EPSILON*math.exp(-2.0*eps_*episode))
                            A = choose_action(S, q_table, epsilon)
                            S_, R = get_env_feedback(S, A, day)  # take action & get next state and reward
                            day+=1    # move to next day
                            q_predict = q_table.loc[S, A]

                            # print(' -->x= ',x , ', day= ',day , '-- action =', A, ' -- S+ =',S_)
                            # import time; time.sleep(0.2)

                            # print(S_)
                            if S_ != 'terminal':

                                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
                            else:
                                q_target = R     # next state is terminal
                                is_terminated = True    # terminate this episode

                            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
                            S = S_  # move to next state

                            update_env(S, episode, step_counter+1)
                            step_counter += 1

                            # update the new sample from the profile
                            if step_counter<len(t_data):
                                idx=np.where(t_data == step_counter)[0]
                                xf=xf_data[idx]
                            else:
                                xf=xf_data[-1]
                                # S = 'terminal'
                                # is_terminated = True    # terminate this episode
                                # cnt_policy=0

                        # print('\n number of days ',step_counter)
                        # clear the console
                        clear()

                        #%% The new learned policy
                        q_policy_new=get_policy(q_table)
                        err_policy=np.nanmean(((q_policy_new - q_policy) ** 2))
                        IMPROVE_POLICY.append(err_policy)
                        #% stopping creteria
                        Min_err_policy=(2/(100*len(ACTIONS)))*POLICY_ERROR
                        err_reduction=err_policy_old-err_policy
                        err_policy_old=err_policy
                        Th_policy = np.abs(POLICY_ERROR*N_STATES)
                        print('\n[',episode,'/(',MAX_EPISODES,'+',eps0,')] Policy difference = ', err_policy,
                              '\n--> Error reduction =', err_reduction,' \tunchanges =',cnt_policy-Th_policy, ' episodes\n--> EPSILON =', epsilon,
                              '\n--> POLICY EVALUATION after ',EPISODES_EVAL,' episodes: MSE =', mse_tracking)

                        if err_reduction ==0.0 :#$POLICY_ERROR:
                            cnt_policy+=1
                        else:
                            cnt_policy=0

                        if cnt_policy>Th_policy:
                            run_learning=False
                            print('\nStop training because of the unchanged policy = ', cnt_policy, '  times!!!.')


                        else:
                            q_policy= q_policy_new

                        if MAX_EPISODES>0 and episode>MAX_EPISODES:
                            run_learning=False
                            print('\n\n\n ==> Stop training because of the maximum iteration of ', MAX_EPISODES, '  iteration is reached!!!.')



                        if episode%EPISODES_EVAL==0:
                            filename='./q_table_weights2.pickle'
                            print('\n\n   --- Save the q-table')

                            with open(filename, 'wb') as f:
                                pickle.dump([q_table, episode, N_STATES, ACTIONS,IMPROVE_POLICY], f)

                            # Evaluation the trained policy
                            q_table = build_q_table(N_STATES, ACTIONS)
                            time, food, fish_weight, total_food, Temp = rl_evaluation(q_table)                 # run policy evaluation on the trained Q-learning table
                            mse_tracking=100*np.nanmean(((xf_data - fish_weight)/fish_weight) ** 2)
                            TRACKING_MSE.append(mse_tracking)
                            TRACKING_MSE_eps.append(episode)



                            # plt.figure(1)
                            # plt.subplot(211)
                            # plt.plot(time, xf_data, '-.k')
                            # plt.plot(time, fish_weight, label='Q-Learning    [ $W_{final}$= '+str(format(fish_weight[-1], '.1f'))+'g, MSE= '+str(mse_tracking)+']')
                            # plt.title('Policy evaluation [Exploitation ='+str(EPISODES_EVAL)+']')
                            # plt.xlabel('Episodes')
                            # plt.ylabel('Relative error(%)')
                            # plt.legend()#loc='center left', bbox_to_anchor=(1, 0.5))
                            #
                            # plt.subplot(212)
                            # plt.plot(time, food, label='Total feed [ $W_{final}$= '+str(format(total_food, '.2f'))+'g]')
                            # plt.title('Feeding policy [Exploitation ='+str(EPISODES_EVAL)+']')
                            # plt.xlabel('Episodes')
                            # plt.ylabel('Percent of body weight \nper day(BWD)')
                            # plt.legend()#loc='center left', bbox_to_anchor=(1, 0.5))
                            #
                            #
                            # #
                            # # plt.figure(2)
                            # # # plt.subplot(312)
                            # # plt.scatter(episode, mse_tracking, label='MSE')
                            # # plt.title('Policy evaluation [Exploitation ='+str(EPISODES_EVAL)+']')
                            # # plt.xlabel('Episodes')
                            # # plt.ylabel('Relative error(%)')
                            # # plt.legend()
                            # plt.pause(0.00001)
                    #
                    #
                    #     plt.subplot(313)
                    #     plt.scatter(episode, err_policy)
                    #     plt.title('Policy improvement [Exploration]')
                    #     plt.xlabel('Episodes')
                    #     plt.ylabel('Policy error')
                    #
                    # plt.savefig('./results2/iter_policy_eps'+str(episode)+suff+'.png', format='png', dpi=1200)
                    # # plt.show()

                    return q_table, episode

                def rl_evaluation(q_table):
                    global x, xf, xf_data
                    episode=-1
                    t = 0
                    S = 0
                    day=0
                    x=x0
                    is_terminated = False
                    update_env(S, episode, t)
                    fish_weight=[]
                    time=[]
                    food=[]
                    Temp=[]

                    min_feeding=False

                    while not is_terminated:
                        if min_feeding== False:
                            # get the Q-learning  action from Q-table
                            # state_actions = q_table.iloc[S, :]
                            # A = state_actions.idxmax()
                            A=choose_action(S, q_table, 1.0)
                        else:
                            A=ACTIONS[0]

                        fish_weight.append(x)
                        time.append(t)
                        food.append(ACTIONS_value[ACTIONS.index(A)][0])
                        Temp.append(ACTIONS_value[ACTIONS.index(A)][1])

                        S_, R = get_env_feedback(S, A, day)  # take action & get next state and reward
                        day+=1    # move to next day
                        if S_ == 'terminal':
                            min_feeding=True

                        if t==len(xf_data)-1:#    S_ == 'terminal':
                            is_terminated = True    # terminate this episode
                            interaction = '\n  -- Total days = %d \n  -- Final fish weight = %.3f / %.3f' % (t, fish_weight[-1], xf_data[-1])
                            print('\r{}'.format(interaction), end='')

                        S = S_  # move to next state
                        # update_env(S, episode, t+1)
                        t += 1

                        # time, food, fish_weight = np.asarray(time), np.asarray(food), np.asarray(fish_weight)
                        total_food=dt*np.sum(R_max*(np.multiply(food,fish_weight)))
                    return  np.asarray(time), np.asarray(food), np.asarray(fish_weight), total_food, np.asarray(Temp)

                ###################################################################################################################
                #%% descritizing the weights
                x_float = np.linspace(x0,xf,W_STATES)
                x_int   = np.digitize(x_float,x_float)-1

                # descritizing the times
                t_float = np.linspace(t_data[0],t_data[-1],TIME_ZONES)
                t_int   = np.digitize(t_float,t_float)-1
                # print('t_float=',t_float)
                # print('t_int=',t_int)
                x=x0
                ##################################################################################################################
                filename='./results/RL2_'
                #TAGGING the experiment
                suff='_r'+str(reward_)+'_grdy'+str(eps_)+'_alph'+str(alpha)+'_gamma'+str(gamma)+'_zh'+str(ZH_)+'_Lz'+str(L_ZH)+'_x0_'+str(math.floor(x0))+'_xf'+str(math.floor(xf))#+'_RMSE'+str(RMSE)    # tagging  the experiemnt
                filename_q_table='./q_table_weights2.pickle'
                if RESET_RL==1 and os.path.isfile(filename_q_table):
                    os.remove('q_table_weights2.pickle')
                #%%  Run the environment
                q_table, episode = rl(N_STATES, ACTIONS)

                ##################   Optimal policy evaluation     ########################
                print('\n-->Optimal policy evaluation  \n')
                # q_table = build_q_table(N_STATES, ACTIONS)
                time, food, fish_weight, total_food, Temp = rl_evaluation(q_table)                 # run policy evaluation on the trained Q-learning table

                mse_tracking=100*np.nanmean(((xf_data - fish_weight)/fish_weight) ** 2)

                suff='mse'+str(math.floor(mse_tracking*10)/10)+'_eps'+str(episode)+suff+'_xt'+str(math.floor(fish_weight[-1]*100)/100)
                print('\n  -- MSE=',mse_tracking)

                ##################   Plot the performcnce     ########################
                plot_and_save_results(time, food, fish_weight, Temp, IMPROVE_POLICY, filename, suff)


print('############   THE END    #############')
# sys.exit()
