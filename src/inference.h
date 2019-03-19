#ifndef INFERENCE_H_
#define INFERENCE_H_

#include <vector>
#include <map>
#include "./layer.h"
#include "./param.h"
#include "./tensor.h"
#include "./tools/parse_binary.h"
namespace tinycnn {

class Inference {
public:
	typedef std::map<int, std::vector<Tensor*>> ALLDATA;

	Inference() {
		//parse();
	}

  //inference
  void inference(){}

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
	//std::vector<Tensor*> all_input_;
	ALLDATA all_data_;
	std::vector<LayerParams*> all_layer_params_;
	std::vector<Layer*> all_layers_;
};

}

#endif
