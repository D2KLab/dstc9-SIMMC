




def print_sample_dialogue(dialogue, annotations=True):
    """Print an annotated sample of the specified dialogue

    Args:
        dialogue (list): list of dialogue turns
    """
    for dial in dialogue:
        print('+U{}: {}\n+A{}: {}'.format(dial['turn_idx'].item(), dial['transcript'], dial['turn_idx'].item(), dial['system_transcript']))
        if annotations:
            print('------- Annotations: turn{}--------'.format(dial['turn_idx'].item()))
            print('+belief_state:{}\n+transcript_annotated{}\n+system_transcript_annotated{}\n+turn_label{}\n+state_graph_0:{}\n+state_graph_1:{}\n+state_graph_2:{}'
                    .format(dial['belief_state'], dial['transcript_annotated'], dial['system_transcript_annotated'], dial['turn_label'], 
                    dial['state_graph_0'], dial['state_graph_1'], dial['state_graph_2']))
            print('-------------------------------\n\n')