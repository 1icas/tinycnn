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

int main() {
  {

  }
}