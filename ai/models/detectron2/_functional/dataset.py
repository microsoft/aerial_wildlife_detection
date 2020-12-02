'''
    Generic function that prepares AIDE's data format
    and converts it into a Detectron2-compliant sibling.
    Note that this function does not load images or checks
    if they are corrupt; this is done at runtime by the
    Detectron2DatasetMapper.

    2020 Benjamin Kellenberger
'''

from detectron2.structures import BoxMode


def getDetectron2Data(aideData, ignoreUnsure=False, filterEmpty=False):
    # create labelclassMap first
    labelclassMap = {}
    idx = 0
    for lcID in aideData['labelClasses']:
        if lcID not in labelclassMap:
            labelclassMap[lcID] = idx       #NOTE: we do not use the labelclass' serial 'idx', since this might contain gaps
            idx += 1

    # parse over entries
    dataset_dicts = []
    unknownClasses = set()
    for idx, key in enumerate(aideData['images']):
        nextMeta = aideData['images'][key]

        record = {
            'file_name': nextMeta['filename'],
            'image_id': idx,
            'image_uuid': key
        }
        annotations = []
        if 'annotations' in nextMeta:
            for anno in nextMeta['annotations']:
                unsure = (anno['unsure'] if 'unsure' in anno else False)
                if unsure and ignoreUnsure:
                    continue

                obj = {}
                if 'x' in anno and 'y' in anno \
                    and 'width' in anno and 'height' in anno:
                    # bounding box
                    obj['bbox'] = [
                        anno['x'],
                        anno['y'],
                        anno['width'],
                        anno['height']
                    ]
                    obj['bbox_mode'] = BoxMode.XYWH_REL     # not yet supported by Detectron2, but by Detectron2DatasetMapper
                elif 'segmentationmask' in anno:
                    # pixel-wise segmentation mask
                    obj['segmentationMask'] = anno['segmentationmask']
                elif 'x' in anno and 'y' in anno:
                    # point (not yet supported by Detectron2)
                    continue
                
                if 'label' in anno:
                    if anno['label'] not in labelclassMap:
                        # unknown label class; ignore for now
                        unknownClasses.add(anno['label'])
                        continue
                    obj['category_id'] = labelclassMap[anno['label']]
                annotations.append(obj)
        
        if filterEmpty and not len(annotations):
            # no annotations in image; skip
            continue

        record['annotations'] = annotations
        dataset_dicts.append(record)
    
    if len(unknownClasses):
        print('WARNING: encountered unknown label classes: {}'.format(', '.join(list(unknownClasses))))
    
    return dataset_dicts