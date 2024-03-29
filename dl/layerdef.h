#ifndef LAYERDEF_H
#define LAYERDEF_H

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

class FcLayer;
class SoftmatLayer;
class ResidualLayer;
class BatchNorm1d;
class LSTM;
class Conv2d;
class MaxPooling2d;
class AvgPooling2d;
class ResidualConv2d;
class NMS;
#endif // LAYERDEF_H
