#include "config.h"
#include "error_handler.h"

namespace tinycnn {

int get_num_bytes(const Type& type) {
  switch(type) {
    case Type::t_char8:
    case Type::t_uchar8:
      return 1;
    case Type::t_short16:
		case Type::t_ushort16:
      return 2;
    case Type::t_float32:
    case Type::t_int32:
		case Type::t_uint32:
      return 4;
    case Type::t_double64:
      return 8;

    default:
      NOT_SUPPORT;
  }
}

Type index2type(const int index) {
	switch(index) {
		case 0:
			return Type::t_char8;
		case 1:
			return Type::t_uchar8;
		case 2:
			return Type::t_short16;
		case 3:
			return Type::t_ushort16;
		case 4:
			return Type::t_float32;
		case 5:
			return Type::t_int32;
		case 6:
			return Type::t_uint32;
		case 7:
			return Type::t_double64;
		default:
			NOT_SUPPORT;
	}
}



}


