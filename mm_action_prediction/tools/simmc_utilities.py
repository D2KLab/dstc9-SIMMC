import os
import sys

import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass  


def plotting_loss(save_path, x_values, x_label, y_label, plot_title, functions, legend=True):
    """plot functions

    Args:
        save_path (str): path where to save the plot
        x_values (numpy.array): values on the x axis
        x_label (str): label for the x axis
        y_label (str): label for the y axis
        plot_title (str): title for the plot
        functions (list): list of tuples (list(values), color, label) where color and label are strings
        legend (bool): to print the legend for the plot. (Default: True)
    """

    # plot train vs validation
    for f in functions:
        plt.plot(x_values, f[0], color=f[1], label=f[2])
    
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend:
        plt.legend(loc='best')
    plt.savefig(save_path)
    plt.clf()


def print_annotation_dialogue(dialogue, actions):
    """Print the specified dialogue with belief state and state graph annotations

    Args:
        dialogue (list): list of dialogue turns
        actions (list): list of actions with shape [ {'turn_idx': <idx>, 'action': <action>, 'action_supervision': {'attributes': [...]}}, ...]
    """
    """
    for turn, act in zip(dialogue, actions):
        print('+U{}: {} -> {}({})'.format(turn['turn_idx']).item(), turn['transcript'])
        #TODO end this function
    """

    for turn in dialogue:    
        print('+U{}: {}\n+A{}: {}'.format(turn['turn_idx'], turn['transcript'], turn['turn_idx'], turn['system_transcript']))
        print('------- Annotations: turn{}--------'.format(turn['turn_idx']))
        print('+belief_state:{}\n+transcript_annotated{}\n+system_transcript_annotated{}\n+turn_label{}\n+state_graph_0:{}\n+state_graph_1:{}\n+state_graph_2:{}'
                    .format(turn['belief_state'], turn['transcript_annotated'], turn['system_transcript_annotated'], turn['turn_label'], 
                    turn['state_graph_0'], turn['state_graph_1'], turn['state_graph_2']))
        print('-------------------------------\n\n')
        


def print_sample_dialogue(dialogue, annotations=True):
    """Print an annotated sample of the specified dialogue

    Args:
        dialogue (list): list of dialogue turns
    """
    for turn in dialogue:
        print('+U{}: {}\n+A{}: {}'.format(turn['turn_idx'].item(), turn['transcript'], turn['turn_idx'].item(), turn['system_transcript']))
        if annotations:
            print('------- Annotations: turn{}--------'.format(turn['turn_idx'].item()))
            print('+belief_state:{}\n+transcript_annotated{}\n+system_transcript_annotated{}\n+turn_label{}\n+state_graph_0:{}\n+state_graph_1:{}\n+state_graph_2:{}'
                    .format(turn['belief_state'], turn['transcript_annotated'], turn['system_transcript_annotated'], turn['turn_label'], 
                    turn['state_graph_0'], turn['state_graph_1'], turn['state_graph_2']))
            print('-------------------------------\n\n')
