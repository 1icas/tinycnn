#include "layer_factory.h"
#include "layer/base_conv.h"
#include "layer/depthwise_conv.h"
#include "layer/batch_norm.h"
#include "layer/flatten.h"
#include "layer/input.h"
#include "layer/maxpool.h"
#include "layer/relu.h"
#include "layer/relux.h"
#include "layer/softmax.h"

namespace tinycnn {

Registry& Registry::Get() {
	static Registry registry;
	return registry;
}

void Registry::insert(const int index, Creator creator, ParamsCreator pcreator) {
	insert_layer(index, creator);
	insert_params(index, pcreator);
}

void Registry::insert_layer(const int index, Creator creator) {
	if (layer_container_.find(index) != layer_container_.end())
		return;
	layer_container_[index] = creator;
}

Layer* Registry::create_layer(const int index, LayerParams* params) {
	if (layer_container_.find(index) == layer_container_.end()) {
		printf("not support the layer index: %d\n", index);
		NOT_SUPPORT;
	}
	return layer_container_[index](params);
}

void Registry::insert_params(const int index, ParamsCreator creator) {
	if (params_container_.find(index) != params_container_.end()) {
		return;
	}
	params_container_[index] = creator;
}

LayerParams* Registry::create_params(const int index) {
	if (params_container_.find(index) == params_container_.end()) {
		printf("not support the layer index: %d\n", index);
		NOT_SUPPORT;
	}
	return params_container_[index]();
}

Register(BASECONV_LAYER, BaseConv);
Register(DEPTHWISE_CONV_LAYER, DepthwiseConv);
Register(BATCH_NORM_LAYER, BatchNorm);
Register(FLATTEN_LAYER, Flatten);
Register(INPUT_LAYER, Input);
Register(MAXPOOL2D_LAYER, MaxPool);
Register(RELU_LAYER, Relu);
Register(RELUX_LAYER, ReluX);
Register(SOFTMAX_LAYER, Softmax);
}