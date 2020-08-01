#######################
# Functions to run Clock Sync Simulation
#######################

import pandas as pd
import numpy as np
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

def local_increment(clock_freq,timeslot_period,freq_tolerance=.5):

    fmax=clock_freq*(1+freq_tolerance/100) # Max freq per error
    fmin=clock_freq*(1-freq_tolerance/100) # Min freq per error

    rand_macrotick_length = timeslot_period * np.random.uniform(fmin, fmax)
    return rand_macrotick_length


# Error plot function
def sim_plot(df, node_count):
    for n0, n1 in combinations(range(node_count),2):
        plt.scatter(df.index, df[str(n0)]-df[str(n1)]);

    plt.xlabel('Time Slots')
    plt.ylabel('Skew (milliseconds)')
    plt.title('Skews between Clocks')


# Simulation function
def clock_sync_sim(freq_tolerance=.5, adjustment_func=daisy_adj, clock_freq = 40, timeslot_period = 1, node_count=4, sim_length=40, r=2):

    clock_freq = clock_freq*(10**6)
    timeslot_period = timeslot_period*(10**-3)

    nominal_macrotick_length = clock_freq * timeslot_period

    df = pd.DataFrame(index=range(sim_length), columns=['Counter','Node Reporting']+[str(node) for node in range(node_count)])
    df.Counter=df.index*nominal_macrotick_length

    df['Node Reporting'] = df.index%node_count

    nodes_ref = list(range(node_count))
    nodes_ref.insert(0, nodes_ref.pop())

    for i in df.index:
        reporting_node = df.loc[i,'Node Reporting']
        comparison_node = nodes_ref[reporting_node]

        for j in range(node_count):
            if i == 0:
                df.loc[i,str(j)] = 0
            else:
                prev_time = df.loc[i-1,str(j)]
                df.loc[i,str(j)] = prev_time + local_increment(clock_freq,timeslot_period,freq_tolerance)

        df.loc[i,str(comparison_node)] = df.loc[i,str(comparison_node)] + adjustment_func(df, i, reporting_node, comparison_node, r)


    fmax=clock_freq*(1+freq_tolerance/100) # Max freq per error
    fmin=clock_freq*(1-freq_tolerance/100) # Min freq per error

    print("Bounds for microticks per timeslot: ({}, {}) microticks".format(fmin,fmax))

    sim_plot(df, node_count)
    return df
