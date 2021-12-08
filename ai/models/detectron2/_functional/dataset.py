'''
    Generic function that prepares AIDE's data format
    and converts it into a Detectron2-compliant sibling.
    Note that this function does not load images or checks
    if they are corrupt; this is done at runtime by the
    Detectron2DatasetMapper.

    2020-21 Benjamin Kellenberger
'''

import copy
from detectron2.structures import BoxMode


def getDetectron2Data(aideData, labelclassMap, projectToStateMap={}, ignoreUnsure=False, filterEmpty=False):
    # merge mappings of model state and model-to-project specifier
    labelclassMap = copy.deepcopy(labelclassMap)
    if isinstance(projectToStateMap, dict):
        for key in projectToStateMap.keys():
            target = projectToStateMap[key]
            if target in labelclassMap:
                targetIdx = labelclassMap[target]
                labelclassMap[key] = targetIdx

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
                    record['segmentationMask'] = anno['segmentationmask']
                    break
                elif 'x' in anno and 'y' in anno:
                    # point (not yet supported by Detectron2)
                    continue
                elif 'coordinates' in anno:
                    # polygons
                    obj['segmentation'] = [anno['coordinates']]
                    # add bbox as well
                    xCoords = anno['coordinates'][::2]
                    yCoords = anno['coordinates'][1::2]
                    bounds = [
                        min(xCoords), min(yCoords),
                        max(xCoords), max(yCoords)
                    ]
                    obj['bbox'] = [
                        (bounds[0]+bounds[2])/2.0,
                        (bounds[1]+bounds[3])/2.0,
                        bounds[2],
                        bounds[3]
                    ]
                    obj['bbox_mode'] = BoxMode.XYWH_REL
                elif 'label' in anno:
                    # image labels; skip instances and append to base dict
                    if anno['label'] is None:
                        # unlabeled images; ignore
                        continue
                    if anno['label'] not in labelclassMap:
                        # unknown label class
                        unknownClasses.add(anno['label'])
                        continue
                    else:
                        record['gt_label'] = labelclassMap[anno['label']]
                    break
                if 'label' in anno:
                    # labels for instances
                    if anno['label'] not in labelclassMap:
                        # unknown label class
                        unknownClasses.add(anno['label'])
                        continue
                    else:
                        obj['category_id'] = labelclassMap[anno['label']]
                if len(obj):
                    annotations.append(obj)
        
        if filterEmpty and not len(annotations) and not \
            ('segmentationMask' in record or 'gt_label' in record):
            # no annotations in image; skip
            continue
        
        if len(annotations):
            record['annotations'] = annotations
        dataset_dicts.append(record)
    
    if len(unknownClasses):
        print('WARNING: encountered unknown label classes: {}'.format(', '.join([str(u) for u in unknownClasses])))
    
    return dataset_dicts