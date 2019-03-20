#include "./flatten.h"
#include "../error_handler.h"
#include "../macro.h"

namespace tinycnn {

void FlattenLayer::init(ALLDATA* all_data) {
  Layer::init(all_data);
  Tensor* input = (*all_data)[input_index_[0].first][0];
  const std::vector<int> shape = input->get_shape();
  LESS(shape.size(), 2);
  Tensor* output = nullptr;
  if(data_type_ == Type::t_float32) {
    output = new Tensor(data_type_);
    input->shallow_copy(output);
    output_.push_back(output);
    (*all_data).push_back(OUTPUTDATA{output});
  } else {
    NOT_SUPPORT;
  }
  int f_d = shape[0];
  int s = shape[1];
  for(int i = 2; i < shape.size(); ++i) {
    s *= shape[i];
  }
  output->set_shape(std::vector<int>{f_d, s});
}

}