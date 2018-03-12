import numpy as np
import tensorflow as tf

from lucid.misc.io import show
import lucid.modelzoo.vision_models as models
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform


if __name__ == '__main__':

    model = models.InceptionV1()
    model.load_graphdef()
    _ = render.render_vis(model, "mixed4a_pre_relu:476")
