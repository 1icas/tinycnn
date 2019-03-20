
#include "./relux.h"

#include "../config.h"
#include "../error_handler.h"
#include "../macro.h"

namespace tinycnn {

void ReluXLayer::init(ALLDATA* all_data) {
	Layer::init(all_data);
	const std::vector<int> shape = (*all_data)[input_index_[0].first][0]->get_shape();
	if(data_type_ == Type::t_float32) {
		Tensor* output = new Tensor(shape, data_type_);
		output_.push_back(output);
    (*all_data).push_back(OUTPUTDATA{output});	
	} else {
		NOT_SUPPORT;
	}
}

void ReluXLayer::forward() {
	//if(Config::get().mode() == InferenceMode::t_float32) {
	const Tensor* input = (*all_data_)[input_index_[0].first][0]; 
	Tensor* output = output_[0];	
	if(data_type_ == Type::t_float32) {
		const float* input_data = static_cast<float*>(input->data());
		float* output_data = static_cast<float*>(output->mutable_data());
		auto output_data_size = output->size();
		for(int i = 0; i < output_data_size; ++i) {
			output_data[i] = input_data[i] < min_ ? min_ : input_data[i] > max_ ? max_ : input_data[i];
		}
	} else {
		NOT_SUPPORT;
	}
	//} else {
	//	NOT_SUPPORT;
	//}
	
}

}
