from ppad.modeling.framework import KDAD
from ppad.modeling.backbone import VGG16, KDADStudentVGG
from ppad.datasets import ImageFolder
from ppad.datasets.pipelines import Resize
from ppad.modeling.losses import MseDirectionLoss
from ppad.engine import train_model
