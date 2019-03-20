#include <cstdio>
#include "./inference.h"
#include "./error_handler.h"
#include "./layer_factory.h"
#include "./macro.h"

namespace tinycnn {

Tensor* Inference::inference(Tensor* input) {
	if (all_data_.size() == 0) {
		printf("may be not read the model, pls read the model first");
		THROW_ERROR;
	}
	auto* first_layer = all_data_[0][0];
	input->shallow_copy(first_layer);
	auto layer_count = all_layers_.size();
	for (int i = 0; i < layer_count; ++i) {
		all_layers_[i]->forward();
	}
	return all_data_[all_data_.size()-1][0];
}

void Inference::create_layers() {
	auto reg = Registry::Get();
	auto size = layer_index_.size();

	for(int i = 0; i < size; ++i) {
		auto* layer = reg.create_layer(layer_index_[i], all_layer_params_[i]);
		layer->init(&all_data_);
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


void Inference::read_model(const char* model_name) {
	FILE* file = fopen(model_name, "rb");
	if(!file) {
		printf("open %s failed", model_name);
		THROW_ERROR;
	}
	fseek(file, 0, SEEK_END);
	auto lsize = ftell(file);
	rewind(file);
	char* buff = (char*)malloc(sizeof(char)*lsize);
	fread(buff, 1, lsize, file);
	ModelData data;
	data.index = 0;
	data.data = buff;
	init(data);
	fclose(file);
	free(buff);
}


}
