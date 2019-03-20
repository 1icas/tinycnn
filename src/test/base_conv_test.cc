#include <algorithm>
#include <iostream>

#include "../layer/base_conv.h"
#include "../alias.h"
#include "../config.h"
#include "../macro.h"
#include "../tensor.h"
#include "../param.h"
#include "../error_handler.h"
using namespace tinycnn;
namespace test {
class BaseConvLayerTest : public BaseConvLayer {
	public:
		BaseConvLayerTest(LayerParams* param): BaseConvLayer(param) {
		}

		void test(ALLDATA* all_data) {
			init(all_data);
			forward();
		}

};	
}

using namespace tinycnn;
using namespace test;

int output_shape(int len, int stride, int kernel, int pad) {
	int fill = 0;
	if (pad == 0) {
		fill = std::max(len % stride == 0 ? 
												kernel - stride : kernel - (len % stride), 0); 
	} 

	return (len + fill - kernel) / stride + 1;

}

#include <iostream>

int main(int argc, char* argv[]) {
	{
		BaseConvParams* param = new BaseConvParams();
		int kernel_nums = 1;
		int kernel_h = 3;
		int kernel_w = 3;
		int kernel_c = 1;
		int input_h = 5;
		int input_w = 5;
		int input_c = 1;
		auto output_h = output_shape(input_h, 2, 3, 0);
		auto output_w = output_shape(input_w, 2, 3, 0);
		Tensor* weights = new Tensor(std::vector<int>{kernel_nums,kernel_h,kernel_w,kernel_c}, Type::t_float32);
		Tensor* biases = new Tensor(std::vector<int>{kernel_nums}, Type::t_float32);
		//Tensor out(std::vector<int>{1, output_h, output_w, kernel_nums}, Type::t_float32);
		param->set_weight(weights);
		param->set_biases(biases);
		param->set_kernel(kernel_nums, kernel_h, kernel_w, kernel_c);
		param->set_stride(2, 2);
		param->set_padding(0);
		param->set_type(Type::t_float32);
		INDEX id{0, std::vector<int>{0}};
		param->set_input_index(id);
		// float in_[] = {-5, 5, -3, 7, 0, -3, 2, 10, -1, -9, 5, -1, 3, -2, 10, 6, -1, -10, -3, 1, -2, -4, -10, -10, -5, -7, -1, -8, -9, 1, -10, 0, 6, 8, -2, -6, -6, -6, 6, -2, 0, -3, 9, 10, 9, -7, -2, -10, 0, 8, -1, -1, -9, -2, 8, 9, 1, 8, -6, 1, -2, 5, 10, 7, 8, -9, 8, -3, -2, 3, 0, -7, -10, -6, -1, 4, 10, -9, -1, 3, 0, -3, 3, -3, 9, 7, -6, 0, 10, -9, 10, 0, 4, -6, -9, -8, -6, 7, 5, -6, 3, 9, 9, -3, 3, 4, -4, -5, -9, 9, -2, -10, 4, 1, -7, 1, 3, 9, -3, 7, 10, 4, 5, -10, 2, 1, 2, -10, -3, -10, -7, -3, 6, 6, -9, 0, -9, 6, 6, 2, 6, 3, -7, 5, 3, 6, 5};

		// float w_[] = {5, -4, 1, 4, -2, 0, -4, -2, -1, -5, -3, 5, -4, -5, 4, -2, 1, 2, 4, -3, -3, -5, -1, 4, 2, 3, -5, -3, 0, 2, -4, -1, 5, -5, 0, 3, 5, -1, -2, 4, 0, -1, -4, 2, 1, 3, 4, 5, 0, 1, 3, 4, -1, -2, -1, -4, 2, -1, 4, -5, 3, 2, -3, 1, 2, -3, -1, -1, -5, -5, -5, -3, 3, 1, 0, 0, 0, 5, 1, -4, 4, 2, 0, 3, 1, 5, 5, 1, -3, -4, -2, -3, -5, 2, 3, 4, -5, -1, 4, 2, -1, 1, -5, 1, -2, 1, -4, -4};


		Tensor in(std::vector<int>{1, input_h, input_w, input_c}, Type::t_float32);
		float in_[] = {1, 7, 2, -6, 1, -4, -2, -10, -4, -8, -5, -8, -6, -10, 5, 5, 6, 1, 0, -7, 4, -4, 4, -7, 0};
		float w_[] = {3, 5, 4, -1, 3, -3, -2, 5, -5};
		float* w_data = static_cast<float*>(weights->mutable_data());
		float* in_data = static_cast<float*>(in.mutable_data());
		for(int i = 0; i < weights->size(); ++i) {
			w_data[i] = w_[i];
		}	
		for(int i = 0; i < in.size(); ++i) {
			in_data[i] = in_[i];
		}

		ALLDATA all_data;
		OUTPUTDATA output_data;
		output_data.push_back(&in);
		all_data.push_back(output_data);
		BaseConvLayerTest layer1(param);
		layer1.init(&all_data);
		layer1.forward();	
		const float* out_data = (float*)(all_data[1][0]->data());//(float*)(out.data());
		float true_result[] = {-28, -9, -23, -24, -59, -62, 73, 60, -28};
		for(int i = 0; i < output_h*output_w*kernel_nums; ++i) {
			EQUAL(true_result[i], out_data[i]);
		}
		delete param;
	}		
	{
		BaseConvParams* param = new BaseConvParams();
		int kernel_nums = 4;
		int kernel_h = 3;
		int kernel_w = 3;
		int kernel_c = 3;
		int input_h = 7;
		int input_w = 7;
		int input_c = 3;
		auto output_h = output_shape(input_h, 2, 3, 0);
		auto output_w = output_shape(input_w, 2, 3, 0);
		Tensor* weights = new Tensor(std::vector<int>{kernel_nums,kernel_h,kernel_w,kernel_c}, Type::t_float32);
		Tensor* biases = new Tensor(std::vector<int>{kernel_nums}, Type::t_float32);
		//Tensor out(std::vector<int>{1, output_h, output_w, kernel_nums}, Type::t_float32);
		param->set_weight(weights);
		param->set_biases(biases);
		param->set_kernel(kernel_nums, kernel_h, kernel_w, kernel_c);
		param->set_stride(2, 2);
		param->set_padding(0);
		param->set_type(Type::t_float32);
		INDEX id{0, std::vector<int>{0}};
		param->set_input_index(id);
		Tensor in(std::vector<int>{1, input_h, input_w, input_c}, Type::t_float32);
		float in_[] = {-5, 5, -3, 7, 0, -3, 2, 10, -1, -9, 5, -1, 3, -2, 10, 6, -1, -10, -3, 1, -2, -4, -10, -10, -5, -7, -1, -8, -9, 1, -10, 0, 6, 8, -2, -6, -6, -6, 6, -2, 0, -3, 9, 10, 9, -7, -2, -10, 0, 8, -1, -1, -9, -2, 8, 9, 1, 8, -6, 1, -2, 5, 10, 7, 8, -9, 8, -3, -2, 3, 0, -7, -10, -6, -1, 4, 10, -9, -1, 3, 0, -3, 3, -3, 9, 7, -6, 0, 10, -9, 10, 0, 4, -6, -9, -8, -6, 7, 5, -6, 3, 9, 9, -3, 3, 4, -4, -5, -9, 9, -2, -10, 4, 1, -7, 1, 3, 9, -3, 7, 10, 4, 5, -10, 2, 1, 2, -10, -3, -10, -7, -3, 6, 6, -9, 0, -9, 6, 6, 2, 6, 3, -7, 5, 3, 6, 5};
		float w_[] = {5, -4, 1, 4, -2, 0, -4, -2, -1, -5, -3, 5, -4, -5, 4, -2, 1, 2, 4, -3, -3, -5, -1, 4, 2, 3, -5, -3, 0, 2, -4, -1, 5, -5, 0, 3, 5, -1, -2, 4, 0, -1, -4, 2, 1, 3, 4, 5, 0, 1, 3, 4, -1, -2, -1, -4, 2, -1, 4, -5, 3, 2, -3, 1, 2, -3, -1, -1, -5, -5, -5, -3, 3, 1, 0, 0, 0, 5, 1, -4, 4, 2, 0, 3, 1, 5, 5, 1, -3, -4, -2, -3, -5, 2, 3, 4, -5, -1, 4, 2, -1, 1, -5, 1, -2, 1, -4, -4};

		// float in_[] = {1, 7, 2, -6, 1, -4, -2, -10, -4, -8, -5, -8, -6, -10, 5, 5, 6, 1, 0, -7, 4, -4, 4, -7, 0};
		// float w_[] = {3, 5, 4, -1, 3, -3, -2, 5, -5};

		float* w_data = static_cast<float*>(weights->mutable_data());
		float* in_data = static_cast<float*>(in.mutable_data());
		for(int i = 0; i < weights->size(); ++i) {
			w_data[i] = w_[i];
		}	
		for(int i = 0; i < in.size(); ++i) {
			in_data[i] = in_[i];
		}
		ALLDATA all_data;
		OUTPUTDATA output_data;
		output_data.push_back(&in);
		all_data.push_back(output_data);
		BaseConvLayerTest layer1(param);
		layer1.init(&all_data);
		layer1.forward();	
		const float* out_data = (float*)(all_data[1][0]->data());
		float true_result[] = { -73,  -99,  -42, 3, -84,  -11,   29,   58, -150, -132,  -60,  -71, -104, 20, 7, 46, -81,   59, -34, 16, 
		-71,   48,   18,   35, -127,  -57, -154,  -10, -15,   52,  -16,   56, -132, -134,    7,  -55, -30,  -15,   97,   48, -21,    9,  107,  47,
		87,   55,   10,   45,81,   36,  119,  -62, -139,   92,    7,   52, -3,  -12,  -93,   53, 4,   42,  -63,   74};
		for(int i = 0; i < output_h*output_w*kernel_nums; ++i) {
			EQUAL(true_result[i], out_data[i]);
		}
		delete param;
	}

	std::cout << "base conv test pass" << std::endl;

}



