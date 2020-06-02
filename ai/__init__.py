'''
    AI prediction models and rankers (Active Learning criteria)
    are registered here.
    AIDE will look for the following two dicts for models available
    to projects.

    In order to register your own model, provide an entry in the
    appropriate dict with the name, description (optional), the accepted annotation type(s)
    and the prediction type the model yields.

    For example:

        'python.path.filename.MyGreatModel' : {
                            'name': 'My great model',
                            'description': 'This is my great deep detection model, based on <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf" target="_blank">YOLO</a>',
                            'annotationType': [
                                'points',
                                'boundingBoxes'
                            ],
                            'predictionType': 'boundingBoxes'
        }

    This model (named "My great model") is located in /python/pyth/filename.py,
    with "filename.py" having a class called "MyGreatModel".
    As can be seen, the description accepts a few HTML markup commands (scripts and
    other potentially malicious entries are strictly ignored).
    The model accepts *both* points and bounding boxes as annotations (ground truth,
    for training), and yields bounding boxes as predictions.

    Available keywords for 'annotationType' and 'predictionType':
    - labels
    - points
    - boundingBoxes
    - segmentationMasks

    If your model only accepts one annotation type (typical case: the same as the
    prediction type), you can also provide a string as a value for 'annotationType',
    instead of an array.


    Similarly, you can define your own AL criterion in the second dict below.
    For example:

        'python.path.myCriterion.MyALcriterion': {
            'name': 'My new Active Learning criterion',
            'description': 'Instead of focusing on the most difficult samples, we just chill and relax.'
        }


    2019-20 Benjamin Kellenberger
'''


# AI prediction models
PREDICTION_MODELS = {

    # built-ins
    'ai.models.pytorch.labels.ResNet': {
                                            'name': 'ResNet',
                                            'author': '(built-in)',
                                            'description': 'Deep classification model based on <a href="http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf" target="_blank">ResNet</a>.',
                                            'annotationType': 'labels',
                                            'predictionType': 'labels'
                                        },
    'ai.models.pytorch.points.WSODPointModel': {
                                            'name': 'Weakly-supervised point detector',
                                            'author': '(built-in)',
                                            'description': '<a href="http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf" target="_blank">ResNet</a>-based point predictor also working on image-wide labels (presence/absence of classes) by weak supervision. Predicts a grid and extracts points from the grid cell centers. Weak supervision requires a fair mix of images with and without objects of the respective classes. See <a href="http://openaccess.thecvf.com/content_CVPRW_2019/papers/EarthVision/Kellenberger_When_a_Few_Clicks_Make_All_the_Difference_Improving_Weakly-Supervised_CVPRW_2019_paper.pdf">Kellenberger et al., 2019</a> for details.',
                                            'annotationType': ['labels', 'points'],
                                            'predictionType': 'points'
                                        },
    'ai.models.pytorch.boundingBoxes.RetinaNet': {
                                            'name': 'RetinaNet',
                                            'author': '(built-in)',
                                            'description': 'Implementation of the <a href="http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf" target="_blank">RetinaNet</a> object detector.',
                                            'annotationType': 'boundingBoxes',
                                            'predictionType': 'boundingBoxes'
                                        },
    'ai.models.pytorch.segmentationMasks.UNet': {
                                            'name': 'U-Net',
                                            'author': '(built-in)',
                                            'description': '<div>Implementation of the <a href="https://arxiv.org/pdf/1505.04597.pdf" target="_blank">U-Net</a> model for semantic image segmentation.</div><img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" height="400px" />',
                                            'annotationType': 'segmentationMasks',
                                            'predictionType': 'segmentationMasks'
                                        }

    # define your own here
}



# Active Learning models
ALCRITERION_MODELS = {

    # built-ins
    'ai.al.builtins.maxconfidence.MaxConfidence': {
                                            'name': 'Max Confidence',
                                            'author': '(built-in)',
                                            'description': 'Prioritizes predictions based on the confidence value of the highest-scoring class.',
                                            'predictionType': ['labels', 'points', 'boundingBoxes', 'segmentationMasks']
                                        },
    'ai.al.builtins.breakingties.BreakingTies': {
                                            'name': 'Breaking Ties',
                                            'author': '(built-in)',
                                            'description': 'Implementation of the <a href="http://www.jmlr.org/papers/volume6/luo05a/luo05a.pdf" target="_blank">Breaking Ties</a> heuristic (difference of confidence values of highest and second-highest scoring classes).',
                                            'predictionType': ['labels', 'points', 'boundingBoxes', 'segmentationMasks']
                                        }
}