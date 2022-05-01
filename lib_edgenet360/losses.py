"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
"""
from tensorflow.keras import backend as K

BATCH_SIZE = 3

def set_batch_size(bs):
    global BATCH_SIZE
    BATCH_SIZE = bs



def weighted_categorical_crossentropy(_y_true, y_pred):
    y_true = K.cast(K.greater_equal(_y_true, K.variable(1)),'float32')
    #print("y_true:\n", K.eval(y_true))

    weights = K.repeat_elements(K.max(_y_true-1, axis=-1, keepdims=True),12, axis=-1)

    occluded = K.cast(K.not_equal(weights,K.variable(1)),'float32') * K.cast(K.not_equal(weights,K.variable(0)),'float32')
    occupied = K.cast(K.equal(weights,K.variable(1)),'float32')

    ratio = (2. * K.sum(occupied))/K.sum(occluded)

    symbolic_shape = K.shape(_y_true)
    #rand_shape = [symbolic_shape[axis] if shape is None else shape
    #               for axis, shape in enumerate(K.int_shape(_y_true))]

    #rand = K.random_uniform_variable(shape=(BATCH_SIZE,60,36,60,12), low=0, high=1, dtype='float32')
    try:
        rand = K.random_uniform_variable(shape=K.shape(_y_true), low=0, high=1, dtype='float32')
    except:
        rand = K.random_uniform_variable(shape=(1,60,36,60,12), low=0, high=1, dtype='float32')

    #rand = K.random_uniform_variable(shape=K.shape(_y_true)[0], low=0, high=1, dtype='float32')

    w = occupied + (occluded * K.cast(K.less_equal(rand, ratio),'float32'))

    #print("classes:\n", K.eval(K.shape(_y_true))[-1])


    norm_weigths =  w  / (K.mean(w)+K.epsilon())
    #print("norm:\n", K.eval(norm_weigths))

    # scale predictions so that the class probas of each sample sum to 1
    y_pred = (y_pred ) / (K.sum(y_pred, axis=-1, keepdims=True))
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * norm_weigths
    loss = -K.sum(loss, -1)
    return loss


def cat_crossent_nobalance(_y_true, y_pred):
    y_true = K.cast(K.greater_equal(_y_true, K.variable(1)),'float32')

    weights = K.repeat_elements(K.max(_y_true-1, axis=-1, keepdims=True),12, axis=-1)

    weights = K.cast(K.not_equal(weights,K.variable(0)),'float32')

    # scale predictions so that the class probas of each sample sum to 1
    y_pred = (y_pred ) / (K.sum(y_pred, axis=-1, keepdims=True))
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss




def dice_loss(y_true, y_pred):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    # outputs = tl.act.pixel_wise_softmax(network.outputs)
    # dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    smooth = 0.00001

    inse = K.sum(y_pred * y_true)
    l = K.sum(y_pred * y_pred)
    r = K.sum(y_true * y_true)

    dice = (2. * inse + smooth) / (l + r + smooth)

    return 1-dice
