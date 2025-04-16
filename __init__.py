from .livepeer_t2i import LivepeerT2I
from .livepeer_i2i import LivepeerI2I
from .livepeer_i2v import LivepeerI2V
from .livepeer_i2text import LivepeerI2T
from .livepeer_upscale import LivepeerUpscale
from .livepeer_jobgetter import NODE_CLASS_MAPPINGS_JOBGETTER, NODE_DISPLAY_NAME_MAPPINGS_JOBGETTER
from .batch_iterator import BatchIterator
from .batch_info import BatchInfo

NODE_CLASS_MAPPINGS = {
    "LivepeerT2I": LivepeerT2I,
    "LivepeerI2I": LivepeerI2I,
    "LivepeerI2V": LivepeerI2V,
    "LivepeerI2T": LivepeerI2T,
    "LivepeerUpscale": LivepeerUpscale,
    "BatchIterator": BatchIterator,
    "BatchInfo": BatchInfo,
    **NODE_CLASS_MAPPINGS_JOBGETTER,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerT2I": "Livepeer T2I",
    "LivepeerI2I": "Livepeer I2I",
    "LivepeerI2V": "Livepeer I2V",
    "LivepeerI2T": "Livepeer I2T",
    "LivepeerUpscale": "Livepeer Upscale",
    "BatchIterator": "Batch Image Iterator",
    "BatchInfo": "Batch Info",
    **NODE_DISPLAY_NAME_MAPPINGS_JOBGETTER,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./js" # Assuming you might have JS extensions 