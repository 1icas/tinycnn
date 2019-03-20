#include "./softmax.h"
#include "../macro.h"
#include <cmath>

namespace tinycnn {
void SoftmaxLayer::init(ALLDATA* all_data) {
  Layer::init(all_data);
  const std::vector<int> shape = (*all_data)[input_index_[0].first][0]->get_shape();
  if(data_type_ == Type::t_float32) {
    Tensor* output = new Tensor(shape, data_type_);
		output_.push_back(output);
    (*all_data).push_back(OUTPUTDATA{output});	
    temp_ = new Tensor(std::vector<int>{shape[0]}, data_type_);
  } else {
    NOT_SUPPORT;
  }
  axis_ = params_->get_softmax_ptr()->get_axis();
}

void SoftmaxLayer::forward() {
  const Tensor* input = (*all_data_)[input_index_[0].first][0]; 
	Tensor* output = output_[0];
  auto shape = input->get_shape();
  //TODO: now we only support two dimensions input data
  if((axis_ == -1 || axis_ == 1) && shape.size() == 2 ) {
    if(data_type_ == Type::t_float32) {
      const float* input_data = static_cast<float*>(input->data());
      float* output_data = static_cast<float*>(output->mutable_data());
      float* temp_data = static_cast<float*>(temp_->mutable_data());
      int n = shape[0];
      int c = shape[1];
      for(int i = 0; i < n; ++i) {
        float count = 0.;
        int index = i * c;
        for(int j = 0; j < c; ++j) {
          temp_data[i] += exp(input_data[index+j]);
        }
      }
      int index = 0;
     for(int i = 0; i < n; ++i) {
        float count = 0.;
        int index = i * c;
        for(int j = 0; j < c; ++j) {
          output_data[index] = exp(input_data[index]) / temp_data[i];
          ++index;
        }
      } 
    } else {
      NOT_SUPPORT;
    }
  } else {
    NOT_SUPPORT;
  }

}

}