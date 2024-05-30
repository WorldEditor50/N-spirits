#ifndef LAYERDEF_H
#define LAYERDEF_H

enum OperateType {
    OP_INPUT = 0,
    OP_FORWARD,
    OP_CONCAT,
    OP_OUTPUT
};

enum LayerType {
    Layer_FullyConnection = 0,
    Layer_Dropout,
    Layer_Norm,
    Layer_Softmax,
    Layer_Concat,
    Layer_Lstm,
    Layer_Conv2d,
    Layer_MaxPooling2d,
    Layer_AvgPooling2d
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
