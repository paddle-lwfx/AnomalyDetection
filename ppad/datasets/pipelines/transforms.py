import traceback

from paddle.vision.transforms import Resize as p_Resize

from ..registry import PIPELINES
from ..builder import build


@PIPELINES.register()
class Compose(object):
    """
    Composes several pipelines(include decode func, sample func, and transforms) together.

    Note: To deal with ```list``` type cfg temporaray, like:

        transform:
            - Crop: # A list
                attribute: 10
            - Resize: # A list
                attribute: 20

    every key of list will pass as the key name to build a module.
    XXX: will be improved in the future.

    Args:
        pipelines (list): List of transforms to compose.
    Returns:
        A compose object which is callable, __call__ for this Compose
        object will call each given :attr:`transforms` sequencely.
    """

    def __init__(self, pipelines):
        # assert isinstance(pipelines, Sequence)
        self.pipelines = []
        for p in pipelines.values():
            if isinstance(p, dict):
                p = build(p, PIPELINES)
                self.pipelines.append(p)
            elif isinstance(p, list):
                for t in p:
                    # XXX: to deal with old format cfg, ugly code here!
                    temp_dict = dict(name=list(t.keys())[0])
                    for all_sub_t in t.values():
                        if all_sub_t is not None:
                            temp_dict.update(all_sub_t)

                    t = build(temp_dict, PIPELINES)
                    self.pipelines.append(t)
            elif callable(p):
                self.pipelines.append(p)
            else:
                raise TypeError(f'pipelines must be callable or a dict,'
                                f'but got {type(p)}')

    def __call__(self, data):
        for p in self.pipelines:
            try:
                data = p(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform transform [{}] with error: "
                      "{} and stack:\n{}".format(p, e, str(stack_info)))
                raise e
        return data


@PIPELINES.register()
class Resize:
    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation
        self.resize = p_Resize(size)

    def __call__(self, img):
        return self.resize(img)
