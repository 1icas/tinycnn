#include <algorithm>
#include <iostream>

#include "../layer/batch_norm.h"
#include "../alias.h"
#include "../config.h"
#include "../macro.h"
#include "../tensor.h"
#include "../param.h"
#include "../error_handler.h"
using namespace tinycnn;
namespace test {
class BatchNormTest : public BatchNormLayer {
	public:
		BatchNormTest(LayerParams* param): BatchNormLayer(param) {
		}

		void test(ALLDATA* all_data) {
			init(all_data);
			forward();
		}
};	
}

using namespace test;

int main()
{
	{
		BatchNormParams* bnp = new BatchNormParams();
		Tensor in(std::vector<int>{1,3,3,3}, Type::t_float32);
		Tensor* t_mean = new Tensor(std::vector<int>{3}, Type::t_float32);
		Tensor* t_variance = new Tensor(std::vector<int>{3}, Type::t_float32);
		Tensor* t_offset = new Tensor(std::vector<int>{3}, Type::t_float32);
		Tensor* t_scale = new Tensor(std::vector<int>{3}, Type::t_float32);
		bnp->set_mean(t_mean);
		bnp->set_variance(t_variance);
		bnp->set_esp(1e-6);
		bnp->set_scale(t_scale);
		bnp->set_shift(t_offset);
		INDEX id{0, std::vector<int>{0}};
		bnp->set_input_index(id);
		bnp->set_type(Type::t_float32);
		float mean[] = {1,1,2};
		float variance[] = {2,2,2};
		float offset[] = {1,1,1};
		float scale[] = {2,1,1};
		float* m_data = static_cast<float*>(t_mean->mutable_data());
		float* v_data = static_cast<float*>(t_variance->mutable_data());
		float* o_data = static_cast<float*>(t_offset->mutable_data());
		float* s_data = static_cast<float*>(t_scale->mutable_data());
		for(int i = 0; i < t_mean->size(); ++i) {
			m_data[i] = mean[i];
		}
		for(int i = 0; i < t_variance->size(); ++i) {
			v_data[i] = variance[i];
		}
		for(int i = 0; i < t_offset->size(); ++i) {
			o_data[i] = offset[i];
		}
		for(int i = 0; i < t_scale->size(); ++i) {
			s_data[i] = scale[i];
		}

		float* in_data = static_cast<float*>(in.mutable_data());
		float input[] = {0,9,6,6,3,5,5,1,2,7,8,7,2,1,3,0,0,9,1,3,1,5,5,2,0,5,6};
		float output[] = {-0.41421318,6.6568527,3.8284264,8.071066,2.4142132,3.1213198,6.6568527,1.0,1.0,9.485279,5.949746,4.535533,2.4142132,1.0,1.7071066,-0.41421318,0.2928934,5.949746,1.0,2.4142132,0.2928934,6.6568527,3.8284264,1.0,-0.41421318,3.8284264,3.8284264};
		float esp = 1e-6;
		for(int i = 0; i < in.size(); ++i) {
			in_data[i] = input[i];
		}
		ALLDATA all_data;
		OUTPUTDATA output_data;
		output_data.push_back(&in);
		all_data.push_back(output_data);
		BatchNormTest layer1(bnp);
		//layer1.init(&all_data);
		layer1.test(&all_data);

		const float* out_data = (float*)(all_data[1][0]->data());//(float*)(out.data());
		//float true_result[] = {-28, -9, -23, -24, -59, -62, 73, 60, -28};
		for(int i = 0; i < all_data[1][0]->size(); ++i) {
			ALMOST_EQUAL(output[i], out_data[i], 1e-5);
			//std::cout << out_data[i] << std::endl;
		}
		delete bnp;
	}
	{
		float input[] = {0,3,7,8,0,1,5,8,5,0,9,0,9,5,7,8,3,5,5,9,4,8,4,8,3,5,4};
		float mean[] = {1,3,2};
		float variance[] = {2,1,1};
		float offset[] = {2,2,2};
		float scale[] = {1,2,1};
		float output[] = {1.2928951,2.0,6.9999747,6.9497347,-3.99997,1.000005,4.8284197,11.99995,4.999985,1.2928951,13.999941,1.001358e-05,7.6568403,5.9999804,6.9999747,6.9497347,2.0,4.999985,4.8284197,13.999941,3.99999,6.9497347,3.99999,7.99997,3.41421,5.9999804,3.99999};
		float esp = 1e-5;
		BatchNormParams* bnp = new BatchNormParams();
		Tensor in(std::vector<int>{1, 3, 3, 3}, Type::t_float32);
		Tensor* t_mean = new Tensor(std::vector<int>{3}, Type::t_float32);
		Tensor* t_variance = new Tensor(std::vector<int>{3}, Type::t_float32);
		Tensor* t_offset = new Tensor(std::vector<int>{3}, Type::t_float32);
		Tensor* t_scale = new Tensor(std::vector<int>{3}, Type::t_float32);
		bnp->set_mean(t_mean);
		bnp->set_variance(t_variance);
		bnp->set_esp(esp);
		bnp->set_scale(t_scale);
		bnp->set_shift(t_offset);
		INDEX id{ 0, std::vector<int>{0} };
		bnp->set_input_index(id);
		bnp->set_type(Type::t_float32);
		float* m_data = static_cast<float*>(t_mean->mutable_data());
		float* v_data = static_cast<float*>(t_variance->mutable_data());
		float* o_data = static_cast<float*>(t_offset->mutable_data());
		float* s_data = static_cast<float*>(t_scale->mutable_data());
		for (int i = 0; i < t_mean->size(); ++i) {
			m_data[i] = mean[i];
		}
		for (int i = 0; i < t_variance->size(); ++i) {
			v_data[i] = variance[i];
		}
		for (int i = 0; i < t_offset->size(); ++i) {
			o_data[i] = offset[i];
		}
		for (int i = 0; i < t_scale->size(); ++i) {
			s_data[i] = scale[i];
		}

		float* in_data = static_cast<float*>(in.mutable_data());
		for (int i = 0; i < in.size(); ++i) {
			in_data[i] = input[i];
		}
		ALLDATA all_data;
		OUTPUTDATA output_data;
		output_data.push_back(&in);
		all_data.push_back(output_data);
		BatchNormTest layer1(bnp);
		//layer1.init(&all_data);
		layer1.test(&all_data);
		const float* out_data = (float*)(all_data[1][0]->data());
		for (int i = 0; i < all_data[1][0]->size(); ++i) {
			ALMOST_EQUAL(output[i], out_data[i], 1e-5);
		}
		delete bnp;
	}
}