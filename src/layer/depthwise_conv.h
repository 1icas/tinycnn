#ifndef LAYER_DEPTHWISE_CONV_H_
#define LAYER_DEPTHWISE_CONV_H_

#include "./base_conv.h"
#include "../param.h"
#include "../tensor.h"

namespace tinycnn {

class DepthwiseConvLayer : public BaseConvLayer {
public:
	DISABLE_COPY_ADN_ASSIGNMENT(DepthwiseConvLayer);
	DepthwiseConvLayer(LayerParams* params) : BaseConvLayer(params) {}
	
	virtual void forward() override;
//	virtual void parse() override;
//	virtual void init(const Tensor* in) override;


private:
	
	void naive_loop_depthwise_conv_op(const Tensor* input, Tensor* output);

};



}



#endif
