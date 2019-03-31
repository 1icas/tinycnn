#ifndef PARAM_H_
#define PARAM_H_

#include <map>

#include "./alias.h"
#include "./config.h"
#include "./data.h"
#include "./macro.h"
#include "./tensor.h"
#include "./tools/parse_binary.h"

namespace tinycnn {
class InputParams;
class BaseConvParams;
class BatchNormParams;
class DepthwiseConvParams;
class FlattenParams;
class ReluParams;
class ReluXParams;
class SoftmaxParams;
class MaxPoolParams;

class LayerParams {
public:

	virtual ~LayerParams() {}

	virtual void parse(ModelData& data){
		TinyStream ts;
		base_parse(ts, data);
	};

	BaseConvParams* get_base_conv_ptr() {
		return reinterpret_cast<BaseConvParams*>(this);
	}

	DepthwiseConvParams* get_depthwise_conv_ptr() {
		return reinterpret_cast<DepthwiseConvParams*>(this);
	}

	ReluXParams* get_relux_ptr() {
		return reinterpret_cast<ReluXParams*>(this);
	}

	BatchNormParams* get_batch_norm_ptr() {
		return reinterpret_cast<BatchNormParams*>(this);
	}

	InputParams* get_input_ptr() {
		return reinterpret_cast<InputParams*>(this);
	}

	SoftmaxParams* get_softmax_ptr() {
		return reinterpret_cast<SoftmaxParams*>(this);
	}

	MaxPoolParams* get_maxpool_ptr() {
		return reinterpret_cast<MaxPoolParams*>(this);
	}

	INPUTINDEX get_input_index() const {
		return input_index_;
	}

	Type get_type() const {
		return data_type_;
	}

	void set_type(Type type) {
		data_type_ = type;
	}

	void set_input_index(INDEX index) {
		input_index_.push_back(index);
	}

protected:
	void base_parse(TinyStream& ts, ModelData& data) {
		parse_data_type(ts, data);
		//parse_output_shape(ts, data);
		parse_layer_input(ts, data);
	}

	void parse_data_type(TinyStream& ts, ModelData& data) {
		data_type_ = index2type(ts.read<t_int32>(data));
	}
	void to_tensor(ModelData& data, Tensor* t);
	void parse_layer_input(TinyStream& ts, ModelData& data);
	//void parse_output_shape(TinyStream& ts, ModelData& data) {
	//	n_ = ts.read<t_int32>(data);
	//	h_ = ts.read<t_int32>(data);
	//	w_ = ts.read<t_int32>(data);
	//	c_ = ts.read<t_int32>(data);
	//}
	//first : prev layer index
	//second: prev layer output index
	INPUTINDEX input_index_;
	Type data_type_;	
	//n h w c
	//int n_;
	//int h_;
	//int w_;
	//int c_;
};

class InputParams : public LayerParams {
public:
	InputParams(){}

	int	get_num() const {
		return n_;
	}

	int get_width() const {
		return w_;
	}

	int get_height() const {
		return h_;
	}

	int get_channel() const {
		return c_;
	}

	virtual void parse(ModelData& data) override {
		TinyStream ts;
		parse_data_type(ts, data);
		n_ = ts.read<t_int32>(data);
		h_ = ts.read<t_int32>(data);
		w_ = ts.read<t_int32>(data);
		c_ = ts.read<t_int32>(data);	
	}

private:
	//n h w c
	int n_;
	int h_;
	int w_;
	int c_;
};

class BatchNormParams : public LayerParams {
public:
	BatchNormParams(): esp_(1e-5), scale_(nullptr), shift_(nullptr), mean_(nullptr), variance_(nullptr){}

	virtual ~BatchNormParams() {
		DELETE(scale_);
		DELETE(shift_);
		DELETE(mean_);
		DELETE(variance_);
	}	

	virtual void parse(ModelData& data) override;

	inline void set_mean(Tensor* mean) {
		mean_ = mean;
	}

	inline void set_variance(Tensor* variance) {
		variance_ = variance;
	}

	inline void set_esp(float esp) {
		esp_ = esp;
	}

	inline void set_scale(Tensor* scale) {
		scale_ = scale;
	}

	inline void set_shift(Tensor* shift) {
		shift_ = shift;
	}

	inline Tensor* get_mean() {
		return mean_;
	}

	inline Tensor* get_variance() {
		return variance_;
	}

	inline float get_esp() const {
		return esp_;
	}

	inline Tensor* get_scale() const {
		return scale_;
	}

	inline Tensor* get_shift() const {
		return shift_;
	}

private:
	//prevent divide zero
	float esp_;
	Tensor* scale_;
	Tensor* shift_;
	Tensor* mean_;
	Tensor* variance_;
};

class BaseConvParams : public LayerParams {
public:
	BaseConvParams():weights_(nullptr), biases_(nullptr){}
	virtual ~BaseConvParams() {
		DELETE(weights_);
		DELETE(biases_);
	}

	inline void set_weight(Tensor* weights) {
		weights_ = weights;
	}
	
	inline void set_biases(Tensor* biases) {
		biases_ = biases;
	}

	inline void set_kernel(const int num, const int h, const int w, int c) {
		kernel_nums_ = num;
		kernel_h_ = h;
		kernel_w_ = w;
		kernel_c_ = c;
	}

	inline void set_stride(const int h, const int w) {
		stride_h_ = h;
		stride_w_ = w;
	}

	inline void set_padding(const int pad) {
		padding_ = pad;
	}
	
	inline int get_kernel_w() const {
		return kernel_w_;
	}

	inline int get_kernel_h() const {
		return kernel_h_;
	}

	inline int get_kernel_nums() const {
		return kernel_nums_;
	}

	inline int get_kernel_c() const {
		return kernel_c_;
	}

	inline int get_stride_w() const {
		return stride_w_;
	}

	inline int get_stride_h() const {
		return stride_h_;
	}

	inline bool is_use_bias() const {
		return use_bias_;
	}

	inline int get_padding() const {
		return padding_;
	}

	inline Tensor* get_weight() const {
		return weights_;
	}

	inline Tensor* get_biases() const {
		return biases_;
	}

	virtual void parse(ModelData& data) override;
	
private:
	//data_format: NHWC, the data is stored in the order of:[batch, height, width, channel]
	Tensor* weights_;
	Tensor* biases_;
	int kernel_nums_;
	int kernel_h_;
	int kernel_w_;
	int kernel_c_;

	int stride_h_;
	int stride_w_;

	int padding_;
	int use_bias_;
};

class DepthwiseConvParams : public BaseConvParams {
public:
	DepthwiseConvParams() {}
	virtual ~DepthwiseConvParams(){}
};

class ReluXParams : public LayerParams {
public:
	ReluXParams(): min_(0), max_(6) {}
	virtual	~ReluXParams() {}
	virtual void parse(ModelData& data);

	inline int get_min() const {
		return min_;
	}

	inline int get_max() const {
		return max_;
	}

private:
	int min_;
	int max_;
}; 

class ReluParams : public LayerParams {

};

class SoftmaxParams : public LayerParams {
public:
	SoftmaxParams(): axis_(-1){}

	int get_axis() const {
		return axis_;
	}

	void set_axis(int axis) {
		axis_ = axis;
	}

	virtual void parse(ModelData& data) override {
		TinyStream ts;
		base_parse(ts, data);
		axis_ = ts.read<t_int32>(data);
	}

private:
	int axis_;
};

class MaxPoolParams : public LayerParams {
public:
	MaxPoolParams(){}

	virtual void parse(ModelData& data) override {
		TinyStream ts;
		base_parse(ts, data);	
		k_h_ = ts.read<t_int32>(data);
		k_w_ = ts.read<t_int32>(data);
		stride_h_ = ts.read<t_int32>(data);
		stride_w_ = ts.read<t_int32>(data);
		padding_ = ts.read<t_int32>(data);
	}

	int get_k_w() const {
		return k_w_;
	}

	int get_k_h() const {
		return k_h_;
	}

	int get_stride_w() const {
		return stride_w_;
	}

	int get_stride_h() const {
		return stride_h_;
	}

	int get_padding() const {
		return padding_;
	}

	void set_kernel_w(int w) {
		k_w_ = w;
	}

	void set_kernel_h(int h) {
		k_h_ = h;
	}

	void set_stride_w(int w) {
		stride_w_ = w;
	}

	void set_stride_h(int h) {
		stride_h_ = h;
	}

	void set_padding(int pad) {
		padding_ = pad;
	}

private:
	int k_w_;
	int k_h_;
	int stride_w_;
	int stride_h_;
	int padding_;
};

class FlattenParams : public LayerParams {

};



}



#endif

