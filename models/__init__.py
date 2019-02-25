from .pix2pix import Pix2Pix
from .glcic import GLCIC
from .srgan import SRGAN


MODELS = {
    Pix2Pix.name: Pix2Pix,
    GLCIC.name: GLCIC,
    SRGAN.name: SRGAN,
}
