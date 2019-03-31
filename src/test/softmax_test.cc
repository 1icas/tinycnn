#include <algorithm>
#include <iostream>

#include "../layer/softmax.h"
#include "../alias.h"
#include "../config.h"
#include "../macro.h"
#include "../tensor.h"
#include "../param.h"
#include "../error_handler.h"
using namespace tinycnn;
namespace test {
class SoftmaxLayerTest : public SoftmaxLayer {
	public:
		SoftmaxLayerTest(LayerParams* param): SoftmaxLayer(param) {
		}
		void test(ALLDATA* all_data) {
			init(all_data);
			forward();
		}
};	
}

using namespace test;

int main(int argc, char* argv[]) {
  SoftmaxParams* params = new SoftmaxParams();
  params->set_axis(-1);
	INDEX id{ 0, std::vector<int>{0} };
	params->set_input_index(id);
	params->set_type(Type::t_float32);
  Tensor in(std::vector<int>{1, 25}, Type::t_float32);
  float in_[] = {1, 7, 2, -6, 1, -4, -2, -10, -4, -8, -5, -8, -6, -10, 5, 5, 6, 1, 0, -7, 4, -4, 4, -7, 0};
  float* in_data = static_cast<float*>(in.mutable_data());
  for(int i = 0; i < in.size(); ++i) {
    in_data[i] = in_[i];
  }
  ALLDATA all_data;
  OUTPUTDATA output_data;
  output_data.push_back(&in);
  all_data.push_back(output_data);
  SoftmaxLayerTest layer1(params);
  layer1.init(&all_data);
  layer1.forward();	
  const float* out_data = (float*)(all_data[1][0]->data());
  float true_result[] = {0.0014129512,0.57002515,0.0038407992,1.2884447e-06,0.0014129512,9.5203895e-06,7.034669e-05,2.3598684e-08,9.5203895e-06,1.7437202e-07,3.5023556e-06,1.7437202e-07,1.2884447e-06,2.3598684e-08,0.07714451,0.07714451,0.20970054,0.0014129512,0.00051979563,4.739923e-07,0.02837988,9.5203895e-06,0.02837988,4.739923e-07,0.00051979563};
  for(int i = 0; i < in.size(); ++i) {
		ALMOST_EQUAL(true_result[i], out_data[i], 1e-5);
  }
  delete params;
}
