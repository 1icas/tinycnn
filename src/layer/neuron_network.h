#include "../layer.h"
#include "../macro.h"
#include "../param.h"


namespace tinycnn {

class NeuronNetworkLayer : public Layer {
public:
	DISABLE_COPY_ADN_ASSIGNMENT(NeuronNetworkLayer);
	NeuronNetworkLayer(LayerParams* params): Layer(params) {}
	~NeuronNetworkLayer(){}

	virtual const char* layer_type() const override {
		return NEURON_NETWORK_TYPE; 
	}


};


}
