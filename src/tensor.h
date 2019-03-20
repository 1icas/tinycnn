#ifndef TENSOR_H_
#define TENSOR_H_

#include <memory.h>
#include <vector>

#include "config.h"
#include "error_handler.h"
#include "macro.h"

namespace tinycnn {

//naive implemention
//don't consider the byte alignment now
class Tensor {
public:
  DISABLE_COPY_ADN_ASSIGNMENT(Tensor);
  Tensor(Type type):data_(nullptr), size_(0), type_(type){}

  Tensor(const std::vector<int>& shape, Type type);

  ~Tensor();

  void set_shape(const std::vector<int>& shape);

  //void set_type(Type type);
  void shallow_copy(Tensor* t);
 
  inline void* data() const {
    return data_->data;
  }
	
	inline int size() const {
		return size_;
	}

  inline void* mutable_data() {
    return data_->data;
  }

  std::vector<int> get_shape() const {
    return shape_;
  }

private:

  // we must be used the malloc_data and free_data to malloc and free 
  // the 'data_' 
  inline void free_data() {
    if(!data_) {
      return;
    }
    desc_count();
		if(data_->ref_count == 0)
			DELETE(data_);
  }

  inline void malloc_data(int size) {
		auto n = get_num_bytes(type_) * size;	
    void* data = malloc(n);
		memset(data, 0, n);
    if(data == nullptr) {
      THROW_ERROR;
    }
    data_ = new Data();
    data_->data = data;
    incr_count();
  }

  inline int compute_size(const std::vector<int>& shape) {
    if(!shape.size()) return 0;
    int size = shape[0];
    for(int i = 1; i < shape.size(); ++i) {
      size *= shape[i];
    }
    return size;
  }

  inline void incr_count() {
    ++data_->ref_count;
  }

  inline void desc_count() {
    --data_->ref_count;
    EQUAL_OR_GREATER(data_->ref_count, 0);
  }

private:
  struct Data {
  public:
    DISABLE_COPY_ADN_ASSIGNMENT(Data); 
    Data(): ref_count(0), data(nullptr){}
    ~Data() {
      if(ref_count == 0) {
        free(data);
      }
    }

public:
    int ref_count;
    void* data;
  };

private:
  Data* data_;
  std::vector<int> shape_;
  int size_;
  Type type_;
};

}


#endif
