#ifndef TOOLS_PARSE_BINARY_H_
#define TOOLS_PARSE_BINARY_H_

#include "../config.h"
#include "../data.h"

namespace tinycnn {

class TinyStream {
public:

	template<typename dtype>
	dtype read(ModelData& data) {
		auto size = sizeof(dtype);
		dtype* p = static_cast<dtype*>(data.data);
		dtype value = p[data.index / size];
		data.index += size;
		return value;
	}



};


	

}




#endif
