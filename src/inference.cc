#include "./inference.h"
#include <cstdio>
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
