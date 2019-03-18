#include "./activation.h"
#include "../param.h"
#include "../tensor.h"



namespace tinycnn {

class ReluXLayer : public ActivationLayer {
public:
	DISABLE_COPY_ADN_ASSIGNMENT(ReluXLayer);	
	ReluXLayer(LayerParams* params) : ActivationLayer(params) {
		ReluXParams* param = params_->get_relux_ptr();
		min_ = param->get_min();
		max_ = param->get_max();
	}

	virtual void forward() override;
	virtual void init(ALLDATA* all_data) override;

private:
	int min_;
	int max_;
}; 

}

