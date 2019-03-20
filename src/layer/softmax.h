#ifndef LAYER_SOFTMAX_H_
#define LAYER_SOFTMAX_H_

#include "./activation.h"
#include "../macro.h"
#include "../param.h"

namespace tinycnn {

class SoftmaxLayer : public ActivationLayer {
public:
  DISABLE_COPY_ADN_ASSIGNMENT(SoftmaxLayer);
  SoftmaxLayer(LayerParams* params): ActivationLayer(params), axis_(-1), temp_(nullptr) {}
  ~SoftmaxLayer() {
    DELETE(temp_);
  }
  virtual void init(ALLDATA* all_data) override;
  virtual void forward() override;
private:
  //if axis_ == -1, we handle the last dimension value
  int axis_;
  Tensor* temp_;
};

}

#endif