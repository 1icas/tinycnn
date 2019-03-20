#include "./base_conv.h"

#include <algorithm>

#include "../config.h"
#include "../error_handler.h"

namespace tinycnn {

void BaseConvLayer::init(ALLDATA* all_data) {
  Layer::init(all_data);
	BaseConvParams* params = params_->get_base_conv_ptr();
	kernel_h_ = params->get_kernel_h();
	kernel_w_ = params->get_kernel_w();
	kernel_c_ = params->get_kernel_c();
	kernel_nums_ = params->get_kernel_nums();
	stride_h_ = params->get_stride_h();
	stride_w_ = params->get_stride_w();
	use_biases_ = params->get_biases();
	padding_ = params->get_padding();
  const std::vector<int> shape = (*all_data)[input_index_[0].first][0]->get_shape();
	if(padding_ == 0) {
    //the padding mode algorithm reference by
    //https://www.tensorflow.org/versions/r1.10/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
    pad_along_height_ = std::max(shape[1] % stride_h_ == 0 ? 
                            (kernel_h_ - stride_h_) : kernel_h_ - (shape[1] % stride_h_), 0);
    pad_along_width_ = std::max(shape[2] % stride_w_ == 0 ? 
                            (kernel_w_ - stride_w_) : kernel_w_ - (shape[2] % stride_w_), 0);
    pad_top_ = pad_along_height_ / 2;
    pad_bottom_ = pad_along_height_ - pad_top_;
    pad_left_ = pad_along_width_ / 2;
    pad_right_ = pad_along_width_ - pad_left_;
  } else if(padding_ == 1) {
    pad_left_ = 0;
    pad_right_ = 0;
    pad_top_ = 0;
    pad_bottom_ = 0;
  }	
  const int output_w = (shape[2] - kernel_w_ + pad_left_ + pad_right_) / stride_w_ + 1;
  const int output_h = (shape[1] - kernel_h_ + pad_bottom_ + pad_top_) / stride_h_ + 1;
  if(data_type_ == Type::t_float32) {
    Tensor* output = new Tensor(std::vector<int>{shape[0], output_h, output_w, kernel_nums_}, data_type_);
    output_.push_back(output);
    (*all_data).push_back(OUTPUTDATA{output});
  } else {
    NOT_SUPPORT;
  }

}

void BaseConvLayer::forward() {
  naive_loop_convolution_op((*all_data_)[input_index_[0].first][0], output_[0]);
}

void BaseConvLayer::naive_loop_convolution_op(const Tensor* input, Tensor* output) {
	BaseConvParams* params = params_->get_base_conv_ptr();
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
    // const auto f_s = feature_map_size * input_shape[3];
    auto index = 0;
		for(int s_h = -pad_top_; s_h < input_shape[1]+pad_bottom_ && s_h+kernel_h_ < input_shape[1]+pad_along_height_; s_h += stride_h_) {
			for(int s_w = -pad_left_; s_w < input_shape[2]+pad_right_ && s_w+kernel_w_ < input_shape[2]+pad_along_width_; s_w += stride_w_) {
				//const float* input_data1 = input_data + (s_h < 0 ? 0 : s_h >= input_shape[1] ? input_shape[1]-1 : s_h) * input_shape[2]
											//  + (s_w < 0 ? 0 : s_w >= input_shape[2]-1 ? input_shape[2]-1 : s_w);
				for(int n = 0; n < kernel_nums_; ++n) {
				float o = 0;
				// for(int c = 0; c < kernel_c_; ++c) {
					for(int h = 0; h < kernel_h_; ++h) {
						for(int w = 0; w < kernel_w_; ++w) {
							for(int c = 0; c < kernel_c_; ++c) {
								//for 'same' padding. we should skip compute the zero padding region
								if(s_w+w < 0 || s_h+h < 0 || s_w+w >= input_shape[2] || s_h+h >= input_shape[1])  
									continue;
								o += input_data[(s_h+h)*input_shape[2]*input_shape[3] + (s_w+w)*input_shape[3] + c] * weights_params[w_s*n + h*kernel_w_*kernel_c_ + w*kernel_c_ + c];
								// o += input_data[c*feature_map_size + (s_h+h)*input_shape[2] + w + s_w]*weights_params[w_s*n + weights_map_size*c + h*kernel_w_ + w];
							}
							
						}
					}

				o += biases_params[n];
				// }
				output_data[index++] = o;
			}
		}
	}
  } else {
    NOT_SUPPORT;
  }
}

// void BaseConvLayer::im2col(const Tensor* input, Tensor* output) {
//   auto& config = Config::get();
//   if(config.mode_ == InferenceMode::t_float32) {
    
//   } else {
//     NOT_SUPPORT;
//   } 
// }
}
