#ifndef LAYER_MAXPOOL_H_
#define LAYER_MAXPOOL_H_

#include "./activation.h"
#include "../macro.h"
#include "../param.h"

namespace tinycnn {

class MaxPoolLayer : public ActivationLayer {
public:
  DISABLE_COPY_ADN_ASSIGNMENT(MaxPoolLayer);
  MaxPoolLayer(LayerParams* params);

  virtual void init(ALLDATA* all_data) override;
  virtual void forward() override;

private:
  int k_w_;
  int k_h_;
  int stride_w_;
  int stride_h_;
  //0 -> same. 1 -> valid
  int padding_;
  int pad_along_height_;
  int pad_along_width_;
  int pad_left_;
  int pad_right_;
  int pad_bottom_;
  int pad_top_;
};

}

#endif