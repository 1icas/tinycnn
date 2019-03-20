#include "./input.h"
#include "../error_handler.h"
#include "../macro.h"

namespace tinycnn {

void InputLayer::init(ALLDATA* all_data) {
  Layer::init(all_data);
  const InputParams* input_params = params_->get_input_ptr();
  int n = input_params->get_num();
  int w = input_params->get_width();
  int h = input_params->get_height();
  int c = input_params->get_channel();
  Tensor* tensor = new Tensor(std::vector<int>{n,h,w,c}, data_type_);
  output_.push_back(tensor);
  (*all_data).push_back(output_);
}

}