#ifndef ERROR_HANLDER_H_
#define ERROR_HANLDER_H_

#include <cassert>
#include <cmath>

#define NULL_POINTER(pointer) \
  if(pointer == nullptr) assert(false);

#define NOT_SUPPORT \
  assert(false);

#define NOT_EQUAL(v1, v2) \
  assert(v1 != v2);

#define EQUAL(v1, v2) \
  assert(v1 == v2)

#define ALMOST_EQUAL(v1, v2, esp) \
  if(abs(v1-v2) > esp) \
    assert(false)

#define LESS(v1, v2) \
  assert(v1 < v2)

#define EQUAL_OR_GREATER(v1, v2) \
  assert(v1 >= v2)

#define THROW_ERROR \
  assert(false);



#endif
