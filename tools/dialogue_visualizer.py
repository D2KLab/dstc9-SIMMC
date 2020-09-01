import pdb

from simmc_dataset import SIMMCDataset

dataset = SIMMCDataset(data_path='data/simmc_fashion/dev/fashion_dev_dials.json',
                        metadata_path='data/simmc_fashion/fashion_metadata.json')

printed=False
for dial_id, dial in dataset.id2dialog.items():
    coref_map = dial['dialogue_coref_map']
    #inverted_coref = {value: key for key, value in coref_map.items()}
    task_id = dial['dialogue_task_id']
    task = dataset.task_mapping[task_id]
    if printed:
        print('\n\n**********************************\n\n')
    for turn in dial['dialogue']:
        # print only dialogues with memory images
        if not len(task['memory_images']):
            printed=False
            continue
        print('-----------')
        print('+U: {}\n+W: {}\n-V: {}\n@Coref: {}\n*FOC: {}\n*MEM: {}\n*DB: {}\n*KEYSTROKE: {}'.format(
                                turn['transcript'],
                                turn['system_transcript'],
                                turn['visual_objects'],
                                coref_map,
                                task['focus_image'],
                                task['memory_images'],
                                task['database_images'],
                                turn['raw_assistant_keystrokes']))
        print('-----------')
        printed=True
    if printed:
        pdb.set_trace()
