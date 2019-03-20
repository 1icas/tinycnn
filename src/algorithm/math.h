#ifndef ALGORITHM_MATH_H_
#define ALGORITHM_MATH_H_

namespace tinycnn{

template<typename dtype>
void im2col(const dtype* in, dtype* out);

template<typename dtype>
void gemm(const dtype* in, dtype* out);

}










#endif