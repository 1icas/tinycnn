#include <algorithm>
#include <iostream>

#include "../layer/maxpool.h"
#include "../alias.h"
#include "../config.h"
#include "../macro.h"
#include "../tensor.h"
#include "../param.h"
#include "../error_handler.h"
using namespace tinycnn;
namespace test {
class MaxPoolLayerTest : public MaxPoolLayer {
	public:
		MaxPoolLayerTest(LayerParams* param): MaxPoolLayer(param) {
		}
		void test(ALLDATA* all_data) {
			init(all_data);
			forward();
		}
};	
}

using namespace test;

int main(int argc, char* argv[]) {
	{
		MaxPoolParams* params = new MaxPoolParams();
		params->set_kernel_w(3);
		params->set_kernel_h(3);
		params->set_stride_h(2);
		params->set_stride_w(2);
		params->set_padding(0);
		INDEX id{ 0, std::vector<int>{0} };
		params->set_input_index(id);
		params->set_type(Type::t_float32);
		Tensor in(std::vector<int>{1, 7, 7, 1}, Type::t_float32);
		float in_[] = { 2,2,9,8,0,0,6,1,8,3,6,3,6,4,7,8,1,4,8,1,5,8,5,2,0,6,0,6,7,0,6,9,4,3,7,5,4,7,4,9,8,7,3,3,5,5,1,3,1 };
		float* in_data = static_cast<float*>(in.mutable_data());
		for (int i = 0; i < in.size(); ++i) {
			in_data[i] = in_[i];
		}
		ALLDATA all_data;
		OUTPUTDATA output_data;
		output_data.push_back(&in);
		all_data.push_back(output_data);
		MaxPoolLayerTest layer1(params);
		layer1.init(&all_data);
		layer1.forward();
		const float* out_data = (float*)(all_data[1][0]->data());
		float true_result[] = { 8.0,9.0,8.0,6.0,8.0,8.0,8.0,6.0,8.0,9.0,9.0,8.0,5.0,7.0,9.0,8.0 };
		for (int i = 0; i < all_data[1][0]->size(); ++i) {
			ALMOST_EQUAL(true_result[i], out_data[i], 1e-5);
		}
		delete params;
	}
  
	{
		MaxPoolParams* params = new MaxPoolParams();
		params->set_kernel_w(3);
		params->set_kernel_h(3);
		params->set_stride_h(1);
		params->set_stride_w(1);
		params->set_padding(1);
		INDEX id{ 0, std::vector<int>{0} };
		params->set_input_index(id);
		params->set_type(Type::t_float32);
		Tensor in(std::vector<int>{1, 5, 5, 3}, Type::t_float32);
		float in_[] = { 7,5,2,2,1,9,4,0,8,8,8,8,1,1,6,7,5,9,9,4,5,2,9,1,4,9,5,4,9,2,9,3,5,8,2,2,2,7,1,9,8,7,4,1,6,7,4,9,1,0,7,2,9,9,0,7,0,0,4,8,8,6,4,3,3,5,6,1,2,2,0,1,8,6,3 };
		float* in_data = static_cast<float*>(in.mutable_data());
		for (int i = 0; i < in.size(); ++i) {
			in_data[i] = in_[i];
		}
		ALLDATA all_data;
		OUTPUTDATA output_data;
		output_data.push_back(&in);
		all_data.push_back(output_data);
		MaxPoolLayerTest layer1(params);
		layer1.init(&all_data);
		layer1.forward();
		const float* out_data = (float*)(all_data[1][0]->data());
		float true_result[] = { 9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,8.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0 };
		for (int i = 0; i < all_data[1][0]->size(); ++i) {
	//		std::cout << out_data[i] << std::endl;
			ALMOST_EQUAL(true_result[i], out_data[i], 1e-5);
		}
		delete params;
	}
}