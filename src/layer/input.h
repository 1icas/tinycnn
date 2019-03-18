#ifndef LAYER_INPUT_H_
#define LAYER_INPUT_H_

#include "../layer.h"
#include "../macro.h"
#include "../param.h"
#include "../tensor.h"

namespace tinycnn {

class InputLayer : public Layer {
public:
	DISABLE_COPY_ADN_ASSIGNMENT(InputLayer);
	InputLayer(LayerParams* params): Layer(params){}
	virtual const char* layer_type() const override {
		return INPUT_TYPE;
	}
	virtual void init(ALLDATA* all_data) override;
	virtual void forward() override {}
};
}



#endif