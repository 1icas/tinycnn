#include "param.h"

#include <vector>

#include "config.h"
#include "error_handler.h"

namespace tinycnn {

void LayerParams::to_tensor(ModelData& data, Tensor* t) {
	TinyStream ts;
	auto size = t->size();
	for(int i = 0; i < size; ++i) {
		switch(data_type_) {
			case Type::t_float32:
				ts.read<t_float32>(data);
				break;
			case Type::t_int32:
				ts.read<t_int32>(data);
				break;
			case Type::t_char8:
				ts.read<t_char8>(data);
				break;
			case Type::t_uchar8:
				ts.read<t_uchar8>(data);
				break;
			case Type::t_short16:
				ts.read<t_short16>(data);
				break;
			case Type::t_ushort16:
				ts.read<t_ushort16>(data);
				break;
			case Type::t_uint32:
				ts.read<t_uint32>(data);
				break;
			case Type::t_double64:
				ts.read<t_double64>(data);
				break;
			default:
				NOT_SUPPORT;
				break;
		}
	}
}

void LayerParams::parse_layer_input(TinyStream& ts, ModelData& data) {
	//TODO: we change the model format, we use a very easy way to create the model file and parse the model file
	//and then we must be considered the concat layer and split layer, so we change the model file format soon
	//auto input_layer_count = ts.read<t_int32>(data);
	//for(int i = 0; i < input_layer_count; ++i) {
	//	auto input_layer_index = ts.read<t_int32>(data);
	//	auto input_layer_data_count = ts.read<t_int32>(data);
	//	std::vector<int> datas_index;
	//	for(int j = 0; j < input_layer_data_count; ++j) {
	//		datas_index.push_back(ts.read<t_int32>(data));
	//	}
	//	prev_layer_input_[input_layer_index] = datas_index;
	//}


	auto input_layer_count = 1;
	for(int i = 0; i < input_layer_count; ++i) {
		auto input_layer_index = ts.read<t_int32>(data);
		auto input_layer_data_count = 1;
		std::vector<int> datas_index;
		for(int j = 0; j < input_layer_data_count; ++j) {
			datas_index.push_back(0);
		}
		input_index_.push_back(INDEX(input_layer_index, datas_index));
	}
}

void BaseConvParams::parse(ModelData& data) {
	TinyStream ts;
	//if(Config::get().mode_ == InferenceMode::t_float32) {
		//data_type_ = index2type(ts.read<t_int32>(data));	
		base_parse(ts, data);
		kernel_nums_ = ts.read<t_int32>(data);
		kernel_h_ = ts.read<t_int32>(data);
		kernel_w_ = ts.read<t_int32>(data);
		kernel_c_ = ts.read<t_int32>(data);
		stride_h_ = ts.read<t_int32>(data);
		stride_w_ = ts.read<t_int32>(data);
		padding_ = ts.read<t_int32>(data);
		use_bias_ = ts.read<t_int32>(data);

		//w_dim = ts.read<t_int32>(data);
		//std::vector<int> w_shape;
		//for(int i = 0; i < w_dim; ++i) {
		//	w_shape.push_back(ts.read<t_int32>(data));
		//}
		//weights_ = new Tensor(w_shape, data_type);

		//b_dim = ts.read<t_int32>(data);
		//if(b_dim != 0) {
		//	std::vector<int> b_shape;
		//	for(int i = 0; i < b_dim; ++i) {
		//		b_shape.push_back(ts.read<t_int32>(data));
		//	}
		//	biases_ = new Tensor(b_shape, data_type);
		//}

		//int input_layer_count = ts.read<t_int32>(data);
		//std::vector<int> 
		//for(int i = 0; i < input_layer_count; ++i) {
		//
		//}
	
		//auto weight_size = kernel_nums_*kernel_h_*kernel_w_*kernel_c_;
//		auto weight_size = ts.read<t_int32>(data);
		weights_ = new Tensor(std::vector<int>{kernel_nums_, kernel_h_, kernel_w_, kernel_c_}, data_type_);
		if(use_bias_) {
			biases_ = new Tensor(std::vector<int>{kernel_nums_}, data_type_);
		}


//		t_float32* weight_data = static_cast<t_float32*>(weights_->mutable_data());
//		for(int i = 0; i < weight_size; ++i) {
//			weight_data[i] = ts.read<t_float32>(data);	
//		}
//		biases_ = new Tensor(std::vector<int>{kernel_nums_}, Type::t_float32);
//		if(use_bias_) {
//			auto bias_size = ts.read<t_int32>(data);
//			t_float32* bias_data = static_cast<t_float32*>(biases_->mutable_data());
//			for(int i = 0; i < kernel_nums_; ++i) {
//				bias_data[i] = ts.read<t_float32>(data);
//			}
//		}
//	}	else {
//		NOT_SUPPORT;
//	}	
}

void BatchNormParams::parse(ModelData& data) {
	TinyStream ts;
	//if(Config::get().mode_ == InferenceMode::t_float32) {
	base_parse(ts, data);
	auto channel = ts.read<t_int32>(data);
	esp_ = ts.read<t_float32>(data);
	scale_ = new Tensor(std::vector<int>{channel}, data_type_);
	shift_ = new Tensor(std::vector<int>{channel}, data_type_);
	mean_ = new Tensor(std::vector<int>{channel}, data_type_);
	variance_ = new Tensor(std::vector<int>(channel), data_type_);


//		scale_ = ts.read<t_float32>(data);
//		shift_ = ts.read<t_float32>(data);
//		esp_ = ts.read<t_float32>(data);
//		auto mean_size = ts.read<t_int32>(data);
//		mean_ = new Tensor(std::vector<int>{mean_size}, Type::t_float32);
//		t_float32* mean_data = static_cast<t_float32*>(mean_->mutable_data());
//		for(int i = 0; i < mean_size; ++i) {
//			mean_data = ts.read<t_float32>(data);
//		}
//		auto variance_size = ts.read<t_int32>(data);
//		variance_ = new Tensor(std::vector<int>{variance_data});
//		t_float32* variance_data = static_cast<t_float32*>(variance_->mutable_data());
//		for(int i = 0; i < variance_size; ++i) {
//			variance_data[i] = ts.read<t_float32>(data);
//		}
	//} else {
	//	NOT_SUPPORT;
	//}
}

void ReluXParams::parse(ModelData& data) {
	TinyStream ts;
	base_parse(ts, data);
	min_ = ts.read<t_int32>(data);
	max_ = ts.read<t_int32>(data);
}

}
