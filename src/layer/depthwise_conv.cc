#include "./depthwise_conv.h"
#include "../layer_factory.h"
#include "../macro.h"

namespace tinycnn {

void DepthwiseConvLayer::forward() {
	naive_loop_depthwise_conv_op((*all_data_)[input_index_[0].first][0], output_[0]);	
}

void DepthwiseConvLayer::naive_loop_depthwise_conv_op(const Tensor* input, Tensor* output) {
  auto& config = Config::get();
	DepthwiseConvParams* params = params_->get_depthwise_conv_ptr();
	const Tensor* weights = params->get_weight();
	const Tensor* biases = params->get_biases();
  if(data_type_ == Type::t_float32) {
    const auto input_shape = input->get_shape();
    const auto output_shape = output->get_shape();
    const float* input_data = (float*)input->data();
    float* output_data = (float*)output->mutable_data();
    const float* biases_params = (float*)biases->data();
    const float* weights_params = (float*)weights->data();
    const auto weights_shape = weights->get_shape();
    const auto weights_map_size = weights_shape[1] * weights_shape[2];
    const auto w_s = weights_map_size * weights_shape[3];
    const auto feature_map_size = input_shape[1] * input_shape[2];
		const auto offset = kernel_c_;
    // const auto f_s = feature_map_size * input_shape[3];
    auto index = 0;
		for(int s_h = -pad_top_; s_h < input_shape[1]+pad_bottom_ && s_h+kernel_h_ < input_shape[1]+pad_along_height_; s_h += stride_h_) {
			for(int s_w = -pad_left_; s_w < input_shape[2]+pad_right_ && s_w+kernel_w_ < input_shape[2]+pad_along_width_; s_w += stride_w_) {
				//const float* input_data1 = input_data + (s_h < 0 ? 0 : s_h >= input_shape[1] ? input_shape[1]-1 : s_h) * input_shape[2]
											//  + (s_w < 0 ? 0 : s_w >= input_shape[2]-1 ? input_shape[2]-1 : s_w);
				for(int n = 0; n < kernel_nums_; ++n) {
				float o = 0;
				// for(int c = 0; c < kernel_c_; ++c) {
					for(int c = 0; c < kernel_c_; ++c) {
						for(int h = 0; h < kernel_h_; ++h) {
							for(int w = 0; w < kernel_w_; ++w) {
								//for 'same' padding. we should skip compute the zero padding region
								if(s_w+w < 0 || s_h+h < 0 || s_w+w >= input_shape[2] || s_h+h >= input_shape[1])  
									continue;
								o += input_data[(s_h+h)*input_shape[2]*input_shape[3] + (s_w+w)*input_shape[3] + c] * weights_params[w_s*n + h*kernel_w_*kernel_c_ + w*kernel_c_ + c];
								// o += input_data[c*feature_map_size + (s_h+h)*input_shape[2] + w + s_w]*weights_params[w_s*n + weights_map_size*c + h*kernel_w_ + w];
							}
							
						}
						//TODO: which is the depthwise conv incule the biase ?? i don;t know .  must be check
						//o += biases_params[n];
						output_data[index++] = o;
					}
			}
		}
	}
  } else {
    NOT_SUPPORT;
  }
	

}

Register(DEPTHWISE_CONV_LAYER, DepthwiseConv);
}




