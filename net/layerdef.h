#ifndef LAYERDEF_H
#define LAYERDEF_H
#include "activate.h"

enum OperateType {
    OP_INPUT = 0,
    OP_FORWARD,
    OP_CONCAT,
    OP_OUTPUT
};

enum LayerType {
    LAYER_FC = 0,
    LAYER_DROPOUT,
    LAYER_NORM,
    LAYER_SOFTMAX,
    LAYER_CONCAT,
    LAYER_LSTM,
    LAYER_CONV2D,
    LAYER_MAXPOOLING,
    LAYER_AVGPOOLING
};

#endif // LAYERDEF_H
