#include "./batch_norm.h"

#include <cmath>

#include "../config.h"
#include "../error_handler.h"
#include "../layer_factory.h"
#include "../macro.h"

namespace tinycnn {

void BatchNormLayer::init(ALLDATA* all_data) {
	Layer::init(all_data);	
	const std::vector<int> shape = (*all_data)[input_index_[0].first][0]->get_shape();
	if(data_type_ == Type::t_float32) {
		Tensor* output = new Tensor(shape, data_type_);
		output_.push_back(output);
    (*all_data).push_back(OUTPUTDATA{output});
	} else {
		NOT_SUPPORT;
	}
	BatchNormParams* params = params_->get_batch_norm_ptr();
	scale_ = params->get_scale();
	shift_ = params->get_shift();
	esp_ = params->get_esp();
	mean_ = params->get_mean();
	variance_ = params->get_variance();
}

void BatchNormLayer::forward() {
	const Tensor* input = (*all_data_)[input_index_[0].first][0]; 
	Tensor* output = output_[0];
	if(data_type_ == Type::t_float32) {
		const float* input_data = static_cast<float*>(input->data());
		const float* mean_data = static_cast<float*>(mean_->data());
		const float* variance_data = static_cast<float*>(variance_->data());
		const float* scale_data = static_cast<float*>(scale_->data());
		const float* shift_data = static_cast<float*>(shift_->data());
		float* output_data = static_cast<float*>(output->mutable_data());
		auto channel_size = mean_->size();
		auto input_size = input->size();
		
		for(int i = 0; i < input_size; i+=channel_size) {
			const float* a = input_data + i * channel_size;
			float* b = output_data + i * channel_size;
			for(int j = 0; j < channel_size; ++j) {
				b[j] = (a[j] - mean_data[j]) / sqrt(variance_data[j]*variance_data[j] + esp_) * scale_data[j] + shift_data[j];
			}
		}

	} else if(data_type_ == Type::t_uchar8) {
		NOT_SUPPORT;
	}
}

}
