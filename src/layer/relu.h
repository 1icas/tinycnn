#ifndef LAYER_RELU_H_
#define LAYER_RELU_H_

#include "./activation.h"
#include "../param.h"
#include "../tensor.h"

namespace tinycnn {

class ReluLayer : public ActivationLayer {
public:
	DISABLE_COPY_ADN_ASSIGNMENT(ReluLayer);
	ReluLayer(LayerParams* params) : ActivationLayer(params){}
	virtual void init(ALLDATA* all_data) override;
	virtual void forward() override;
	



};
	

}




#endif
