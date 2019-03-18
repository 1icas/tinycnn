#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include "./alias.h"
#include "./config.h"
#include "./error_handler.h"
#include "./macro.h"
#include "./tensor.h"
#include "./param.h"

namespace tinycnn {

class Layer {
public:
	DISABLE_COPY_ADN_ASSIGNMENT(Layer);
  Layer(LayerParams* params): params_(params) {
		data_type_ = params_->get_type();
	}
	
	virtual ~Layer() {
		DELETE(params_);
	}
		
	virtual const char* layer_type () const = 0;
  virtual void forward() = 0;
	virtual void init(ALLDATA* all_data) {
		NULL_POINTER(all_data);
		all_data_ = all_data;
		input_index_ = params_->get_input_index();
		data_type_ = params_->get_type();
	}
  // parse the param for every layer
//  virtual void parse() {}

//  virtual void init(const Tensor* in) {parse();}

//	virtual void set_and_allocate_data(const int index, ALLDATA* all_data_) {
//		if(all_data_ == nullptr) {
//			THROW_ERROR;
//		}
//		index_ = index; 
//		int n = params_->get_num();
//		int h = params_->get_height();
//		int w = params_->get_width();
//		int c = params_->get_channel();
//		if(all_data_->find(index) == all_data_->end()) {
//			std::vector<Tensor*> v;
//			v.push_back(new Tensor(std::vector<int>{n, h, w, c}, params_->get_type()));
//			all_data_->insert(std::pair<int, std::vector<Tensor*>>(index, v));
//		} else {
//			THROW_ERROR;
//		}
//	}
//
protected:
	const LayerParams* get_params() const {
		return params_;
	}

protected:
	LayerParams* params_;
	//std::vector<Tensor*>* all_layer;
	ALLDATA* all_data_;
	Type data_type_;
	INPUTINDEX input_index_;
	OUTPUTDATA output_;
private:

};

}

#endif
