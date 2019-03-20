#ifndef LAYER_FLATTEN_H_
#define LAYER_FLATTEN_H_

#include "./tool.h"

namespace tinycnn {

class FlattenLayer : public ToolLayer {
public:
  DISABLE_COPY_ADN_ASSIGNMENT(FlattenLayer);
  FlattenLayer(LayerParams* params): ToolLayer(params){}
  virtual void forward() override{};
  virtual void init(ALLDATA* all_data) override;
};

}


#endif