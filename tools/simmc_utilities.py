



def print_action_dialogue(dialogue, actions):
    """Print the specified dialogue with action annotations

    Args:
        dialogue (list): list of dialogue turns
        actions (list): list of actions with shape [ {'turn_idx': <idx>, 'action': <action>, 'action_supervision': {'attributes': [...]}}, ...]
    """
    for turn, act in zip(dialogue, actions):
        print('+U{}: {} -> {}({})'.format(turn['turn_idx']).item(), turn['transcript'])
        #TODO end this function

        """
        print('+U{}: {}\n+A{}: {}'.format(turn['turn_idx'].item(), turn['transcript'], turn['turn_idx'].item(), turn['system_transcript']))
        if annotations:
            print('------- Annotations: turn{}--------'.format(turn['turn_idx'].item()))
            print('+belief_state:{}\n+transcript_annotated{}\n+system_transcript_annotated{}\n+turn_label{}\n+state_graph_0:{}\n+state_graph_1:{}\n+state_graph_2:{}'
                    .format(turn['belief_state'], turn['transcript_annotated'], turn['system_transcript_annotated'], turn['turn_label'], 
                    turn['state_graph_0'], turn['state_graph_1'], turn['state_graph_2']))
            print('-------------------------------\n\n')
        """  




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