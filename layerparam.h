#ifndef LAYERPARAM_H
#define LAYERPARAM_H
#include "layerdef.h"

class FcParam
{
public:
    int id;
    int inputDim;
    int outputDim;
    bool bias;
    /* type */
    int opType;
    int activeType;
    int layerType;
public:
    FcParam():inputDim(0),outputDim(0),bias(false),
    opType(OP_FORWARD),activeType(ACTIVE_LINEAR),layerType(LAYER_FORWARD){}
    FcParam(int inDim_, int outDim_, bool bias_, int activeType_):
        inputDim(inDim_),outputDim(outDim_),bias(bias_),
        opType(OP_FORWARD),activeType(activeType_),layerType(LAYER_FORWARD){}
    FcParam(const FcParam &param):
        inputDim(param.inputDim),outputDim(param.outputDim),bias(param.bias),
        opType(param.opType),activeType(param.activeType),layerType(param.layerType){}
};

#endif // LAYERPARAM_H
