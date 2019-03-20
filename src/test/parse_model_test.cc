#include <iostream>
#include "../inference.h"
#include "../layer/depthwise_conv.h"
#include "../layer/relu.h"
#include "../macro.h"
using namespace tinycnn;

int main() {
  Inference inference;
  inference.read_model("../src/test/python/model.p");
	float input[] = { 2, 10, 3.5, -1 };
	float output[] = { 2, 10, 3.5, 0 };
	Tensor in(std::vector<int>{1, 2, 2, 1}, Type::t_float32);
	float* in_data = (float*)(in.mutable_data());
	for (int i = 0; i < in.size(); ++i) {
		in_data[i] = input[i];
	}
	auto* out = inference.inference(&in);
	const float* out_data = (float*)(out->data());
	for (int i = 0; i < out->size(); ++i) {
		ALMOST_EQUAL(out_data[i], output[i], 1e-5);
	}
} 





