#ifndef LAYER_BATCH_NORM_H_
#define LAYER_BATCH_NORM_H_

#include "./neuron_network.h"
#include "../param.h"
#include "../tensor.h"


namespace tinycnn {

class BatchNormLayer : public NeuronNetworkLayer {
public:
	DISABLE_COPY_ADN_ASSIGNMENT(BatchNormLayer);
	BatchNormLayer(LayerParams* params) : NeuronNetworkLayer(params) {}

	virtual void init(ALLDATA* all_data) override;
	virtual void forward() override;

private:
	float esp_;
	Tensor* scale_;
	Tensor* shift_;
	Tensor* mean_;
	Tensor* variance_;
};



}


#endif