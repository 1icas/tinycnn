#ifndef MACRO_H_
#define MACRO_H_

#include <map>
#include <vector>

#define DISABLE_COPY_ADN_ASSIGNMENT(class_name) \
  class_name(const class_name& name) = delete; \
  class_name& operator=(const class_name& name) = delete;


#define DELETE(c) \
	if(c != nullptr) { \
		delete c; \
		c = nullptr; \
  }

#define ACTIVATION_TYPE "act"
#define NEURON_NETWORK_TYPE "neuron_network"
#define INPUT_TYPE "input"
#define TOOL_TYPE "tool"


//define the layer index
//input layer
#define INPUT_LAYER 0

#define BATCH_NORM_LAYER 10

#define BASECONV_LAYER 20
#define DEPTHWISE_CONV_LAYER 21
#define DENSE_LAYER 22


//activation layer
#define RELU_LAYER 60
#define RELUX_LAYER 61
#define MAXPOOL2D_LAYER 62
#define SOFTMAX_LAYER 63

//tool layer
#define FLATTEN_LAYER 80

#endif


