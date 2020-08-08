#######################
# Functions to run Clock Sync Simulation
#######################

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import combinations

# Adjustment functions
def daisy_adj(df, current_index, reporting_node, comparison_node, r=2):
    reporting_node_val = df.loc[current_index,str(reporting_node)]
    comparison_node_val = df.loc[current_index,str(comparison_node)]

    dif = reporting_node_val-comparison_node_val
    return dif/r

def no_adj(df, current_index, reporting_node, comparison_node, r=2):
    return 0

def local_increment(clock_freq,timeslot_period,timeslot_tolerance):
    rand_macrotick_length = timeslot_period * clock_freq*(1 + np.random.uniform(-timeslot_tolerance/100, timeslot_tolerance/100))
    return rand_macrotick_length


# Error plot function
def sim_plot(df, node_count, nominal_tick_count):

    print("Plot Setup Commencing...")

    # Skew Plot
    fig=plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    combs = sum(1 for pair in combinations(range(node_count),2))

    for comb, t in enumerate(combinations(range(node_count),2)):
        n0, n1 = t
        if (comb%(combs//4)) == 0 and comb < combs-4:
            print("{}.0%".format(math.ceil((comb/combs)*100)), end =" - ")
        plt.scatter(df.index, df[str(n0)]-df[str(n1)])

    plt.plot([0,df.index.max()], [nominal_tick_count,nominal_tick_count])
    plt.plot([0,df.index.max()], [-nominal_tick_count,-nominal_tick_count])
    plt.xlabel('Time Slots')
    plt.ylabel('Skew (ticks)')
    plt.title('Skews between Clocks')

    # Time Plot
    fig2=plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(df.index,df['Nominal Counter']-df['Nominal Counter'])
    for node in range(node_count):
        plt.scatter(df.index, df[str(node)]-df['Nominal Counter'])

    fig3=plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(df.index,df['Nominal Counter'])
    for node in range(node_count):
        plt.scatter(df.index, df[str(node)])

    # plt.xlabel('Macroticks (10,000 Microticks)')
    # plt.ylabel('Error')
    # plt.title('X')

    print("100.0%") #Plot Setup Complete...
    print("Please wait a moment for the plot to appear.\n")


# Simulation function
def clock_sync_sim(freq_tolerance=.5, timeslot_tolerance=.5, adjustment_func=daisy_adj, clock_freq = 40, timeslot_period = 1, node_count=4, sim_length=40, r=2):

    # Convert to correct units
    clock_freq = clock_freq*(10**6)
    timeslot_period = timeslot_period*(10**-3)

    # Create Data Frame for Storage with nominal values for comparison
    df = pd.DataFrame(index=range(sim_length), columns=['Nominal Counter','Node Reporting']+[str(node) for node in range(node_count)])
    nominal_macrotick_length = clock_freq * timeslot_period
    df['Nominal Counter']=df.index*nominal_macrotick_length

    # Node Setup
    df['Node Reporting'] = df.index%node_count # Models Schedule of Node Communication
    nodes_ref = list(range(node_count))

    node_clock_freqs = [] # Models pre-set frequency drifts for each node
    for node in range(node_count):
        node_clock_freq = clock_freq * (1 + np.random.uniform(-freq_tolerance/100,freq_tolerance/100))
        node_clock_freqs.append(node_clock_freq)

    print("Simulation Setup Commencing...")
    for i in df.index:
        reporting_node = df.loc[i,'Node Reporting']
        comparison_nodes = nodes_ref.copy()
        comparison_nodes.remove(reporting_node)

        if (i%(sim_length//4)) == 0 and i < sim_length-4:
            print("{}.0%".format(math.ceil((i/sim_length)*100)), end =" - ")

        for node in range(node_count):
            if i == 0:
                df.loc[i,str(node)] = 0
            else:
                prev_time = df.loc[i-1,str(node)]
                df.loc[i,str(node)] = prev_time + local_increment(node_clock_freqs[node],timeslot_period,timeslot_tolerance)

        for comparison_node in comparison_nodes:
            df.loc[i,str(comparison_node)] = df.loc[i,str(comparison_node)] + adjustment_func(df, i, reporting_node, comparison_node, r)

    print("100.0%\n") #Simulation Setup Complete

    fmax=clock_freq*(1+freq_tolerance/100) # Max freq per error
    fmin=clock_freq*(1-freq_tolerance/100) # Min freq per error

    # max_dev = rho(max drift rate) * R(resync interval) * 2 * r
    rho = (fmax-fmin)/clock_freq
    max_dev = rho*timeslot_period*(10**6)*r # Convert to usec
    nominal_tick_count = max_dev*clock_freq/(10**6) # Divide by unit adjustment

    sim_plot(df, node_count, nominal_tick_count)
    print("Maximum Deviation: +/- {:.2f} usec ({} ticks at nominal frequency)".format(max_dev,math.ceil(nominal_tick_count)))
    return df
