#ifndef INFERENCE_H_
#define INFERENCE_H_

#include <vector>
#include <map>
#include "./layer.h"
#include "./alias.h"
#include "./param.h"
#include "./tensor.h"
#include "./tools/parse_binary.h"

namespace tinycnn {

class Inference {
public:

	Inference() {
	
	}

	~Inference() {
		 for(int i = 0; i < all_data_.size(); ++i) {
		 	auto data_group = all_data_[i];
		 	for(int j = 0; j < data_group.size(); ++j) {
		 		auto* t = data_group[j];
		 		DELETE(t);
		 	}
		 }
		for(int i = 0; i < all_layer_params_.size(); ++i) {
			auto* p = all_layer_params_[i];
			DELETE(p);
		}
		for(int i = 0; i < all_layers_.size(); ++i) {
			auto* l = all_layers_[i];
			DELETE(l);
		}
	}

  //inference
	Tensor* inference(Tensor* input);

	void read_model(const char* model_name);

private:
	void init(ModelData& data) {
		parse_layers_index(data);
		create_layer_params();
		parse_layer_params(data);
		create_layers();

	}
  //parse the model file
	//: get all of the layer index
  void parse_layers_index(ModelData& data){
		TinyStream ts;
		auto layer_count = ts.read<t_int32>(data);
		for(int i = 0; i < layer_count; ++i) {
			layer_index_.push_back(ts.read<t_int32>(data));
		}
	}

	void parse_layer_params(ModelData& data);

	void create_layers();

	void create_layer_params();

private:
	std::vector<int> layer_index_;	
	ALLDATA all_data_;
	std::vector<LayerParams*> all_layer_params_;
	std::vector<Layer*> all_layers_;
};

}

#endif
