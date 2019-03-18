#ifndef CONFIG_H_
#define CONFIG_H_

#include "macro.h"

namespace tinycnn {

typedef char t_char8;
typedef unsigned char t_uchar8;
typedef short t_short16;
typedef unsigned short t_ushort16;
typedef int t_int32;
typedef unsigned int t_uint32;
typedef float t_float32;
typedef double t_double64;


enum class Type {
  t_char8,
  t_uchar8,
  t_short16,
	t_ushort16,
  t_float32,
  t_int32,
	t_uint32,
  t_double64
};

int get_num_bytes(const Type& type);
Type index2type(const int index);

enum class InferenceMode {
  t_char,
  t_float32
};

class Config {

public:
  DISABLE_COPY_ADN_ASSIGNMENT(Config);

  static Config& get() {
    static Config config;
    return config;
  }

  InferenceMode mode_;



private:
  Config() {}
  ~Config() {}



};


};



#endif
