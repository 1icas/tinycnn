#ifndef LAYER_ACTIVATION_H_
#define LAYER_ACTIVATION_H_

#include "../layer.h"
#include "../macro.h"
#include "../param.h"

namespace tinycnn {

class ActivationLayer : public Layer {
public:
	DISABLE_COPY_ADN_ASSIGNMENT(ActivationLayer);
	ActivationLayer(LayerParams* params):Layer(params) {}
	~ActivationLayer() {}
	
	virtual const char* layer_type() const override {
		return ACTIVATION_TYPE;
	}
};


}

#endif