#include "./maxpool.h"
#include "./util.h"
#include "../macro.h"
#include<cfloat>

namespace tinycnn {
MaxPoolLayer::MaxPoolLayer(LayerParams* params) :
  ActivationLayer(params) {
  MaxPoolParams* p = params_->get_maxpool_ptr();
  k_h_ = p->get_k_h();
  k_w_ = p->get_k_w();
  stride_h_ = p->get_stride_h();
  stride_w_ = p->get_stride_w();
  padding_ = p->get_padding();
  pad_along_height_ = 0;
  pad_along_width_ = 0;
  pad_top_ = 0;
  pad_bottom_ = 0;
  pad_left_ = 0;
  pad_right_ = 0;
}

void MaxPoolLayer::init(ALLDATA* all_data) {
  Layer::init(all_data);
  const std::vector<int> shape = (*all_data)[input_index_[0].first][0]->get_shape();
  if(padding_ == 0) {
    _pad(shape, k_w_, k_h_, stride_w_, stride_h_, pad_along_width_,
    pad_along_height_, pad_top_, pad_bottom_, pad_left_, pad_right_);
  } else {
    pad_along_height_ = 0;
    pad_along_width_ = 0;
    pad_left_ = 0;
    pad_right_ = 0;
    pad_bottom_ = 0;
    pad_top_ = 0;
  } 
  int output_w = 0;
  int output_h = 0;
  _conv_output(shape[2], shape[1], k_w_, k_h_, pad_left_, pad_right_,
      pad_top_, pad_bottom_, stride_w_, stride_h_, output_w, output_h);
  if(data_type_ == Type::t_float32) {
    Tensor* output = new Tensor(std::vector<int>{shape[0], output_h, output_w, shape[3]}, data_type_);
    output_.push_back(output);
    (*all_data).push_back(OUTPUTDATA{output});
  } else {
    NOT_SUPPORT;
  }
  
}

void MaxPoolLayer::forward() {
  const Tensor* input = (*all_data_)[input_index_[0].first][0]; 
	Tensor* output = output_[0];
  if(data_type_ == Type::t_float32) {
    const auto input_shape = input->get_shape();
    const auto output_shape = output->get_shape();
    const float* input_data = (float*)input->data();
    float* output_data = (float*)output->mutable_data();
    const auto feature_map_size = input_shape[1] * input_shape[2];
    auto index = 0;
		for(int s_h = -pad_top_; s_h < input_shape[1]+pad_bottom_ && s_h+k_h_-1 < input_shape[1]+pad_along_height_; s_h += stride_h_) {
			for(int s_w = -pad_left_; s_w < input_shape[2]+pad_right_ && s_w+k_w_-1 < input_shape[2]+pad_along_width_; s_w += stride_w_) {
				//const float* input_data1 = input_data + (s_h < 0 ? 0 : s_h >= input_shape[1] ? input_shape[1]-1 : s_h) * input_shape[2]
											//  + (s_w < 0 ? 0 : s_w >= input_shape[2]-1 ? input_shape[2]-1 : s_w);
				for(int n = 0; n < input_shape[3]; ++n) {
  				float min = FLT_MIN;
          for(int h = 0; h < k_h_; ++h) {
            for(int w = 0; w < k_w_; ++w) { 
              //   //for 'same' padding. we should skip compute the zero padding region
              if(s_w+w < 0 || s_h+h < 0 || s_w+w >= input_shape[2] || s_h+h >= input_shape[1]) {
                if(min < 0) min = 0;
              } else {
                float v = input_data[(s_h+h)*input_shape[2]*input_shape[3] + (s_w+w)*input_shape[3] + n]; 
                if(min < v) min = v;
              }
            }
          }

				  output_data[index++] = min;
			  }
  		}
  	}
  } else {
    NOT_SUPPORT;
  }
}
}