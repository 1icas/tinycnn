#include "tensor.h"


namespace tinycnn {


Tensor::Tensor(const std::vector<int>& shape, Type type):
  shape_(shape) ,type_(type) {
  //data_ = new Data();
	size_ = compute_size(shape_);
  malloc_data(size_);
	//todo: must be byte alignment
	//data_ = malloc_data(size_);
}

Tensor::~Tensor() {
  free_data();
}

void Tensor::shallow_copy(Tensor* t) {
  if(t->data_ != nullptr) {
    t->free_data();
  }
  t->data_ = data_;
  t->shape_ = shape_;
  t->size_ = size_;
  t->type_ = type_;
  incr_count();
}

void Tensor::set_shape(const std::vector<int>& shape) {
  int size = compute_size(shape);
  shape_ = shape;
  if(size == size_) {
    return;
  }
  size_ = size;
  free_data();
  malloc_data(size_);
}

//void Tensor::set_type(Type type) {
//  type_ = type;
//}


};
