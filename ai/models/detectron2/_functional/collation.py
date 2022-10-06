'''
    Custom collate_fn for Detectron2 that discards invalid entries (i.e., with
    corrupt images).

    2022 Benjamin Kellenberger
'''

def collate(batch):
    batch = [b for b in batch if b.get('image', None) is not None]
    return batch