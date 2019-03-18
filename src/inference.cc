#include "./inference.h"
#include "./layer_factory.h"

namespace tinycnn {

void Inference::create_layers() {
	auto reg = Registry::Get();
	auto size = layer_index_.size();

	for(int i = 0; i < size; ++i) {
		auto* layer = reg.create_layer(layer_index_[i], all_layer_params_[i]);
		all_layers_.push_back(layer);
	}

}

void Inference::create_layer_params() {
	auto reg = Registry::Get();
	auto size = layer_index_.size();
	for(int i = 0; i < size; ++i) {
		auto* layer_params = reg.create_params(layer_index_[i]);
		all_layer_params_.push_back(layer_params);	
	}
}

void Inference::parse_layer_params(ModelData& data) {
	for(auto* p : all_layer_params_) {
		p->parse(data);
	}	
}


}
