#include "../layer.h"
#include "../macro.h"
#include "../param.h"

namespace tinycnn {

class ToolLayer : public Layer {
public:
	DISABLE_COPY_ADN_ASSIGNMENT(ToolLayer);
	ToolLayer(LayerParams* params):Layer(params) {}
	~ToolLayer() {}
	
	virtual const char* layer_type() const override {
		return TOOL_TYPE;
	}
};


}
