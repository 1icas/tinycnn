#ifndef BASE_CONV_H_
#define BASE_CONV_H_

#include "./neuron_network.h"
#include "../layer_factory.h"
#include "../macro.h"
#include "../param.h"
#include "../tensor.h"

namespace tinycnn {

// naive implement for the convolution layer
class BaseConvLayer : public NeuronNetworkLayer {
public:
  DISABLE_COPY_ADN_ASSIGNMENT(BaseConvLayer);
  BaseConvLayer(LayerParams* params): NeuronNetworkLayer(params) {}
  virtual void forward() override;
  virtual void init(ALLDATA* all_data) override;

private:
  void naive_loop_convolution_op(const Tensor* input, Tensor* output);
  //the same convolution algorithm as caffe
  //1. rearrange the feature map (im2col)
  //2. gemm(general matrix to matrix multiplication)
  void caffe_convolution_op(const Tensor* input, Tensor* output);
  template<typename T>
  void im2col(const T* in, T* out);
  void gemm();

protected:
  int kernel_h_;
  int kernel_w_;
  int kernel_nums_;
  int kernel_c_;
  int stride_w_;
  int stride_h_;
  int pad_along_height_ = 0;
  int pad_along_width_ = 0;
  int pad_left_ = 0;
  int pad_right_ = 0; 
  int pad_top_ = 0;
  int pad_bottom_ = 0;
  // 0 -> same  1-> valid
  //TODO: if we want to compatible the tensorflow / pytorch / caffe model and so on. we must
  // be considered the padding strategy
  int padding_;

  //the kernel arrangement is NCHW
	//Tensor weights_;
  //Tensor biases_;
  bool use_biases_;
};


};

#endif

