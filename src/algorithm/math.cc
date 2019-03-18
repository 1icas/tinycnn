#include "math.h"

namespace tinycnn {

template<typename dtype>
void im2col(const dtype* in, 
            const int channel, 
            const int width, 
            const int height, 
            const int kernel_w, 
            const int kernel_h, 
            const int pad_top, 
            const int pad_bottom, 
            const int pad_left, 
            const int pad_right, 
            dtype* out) {
  // for(int c = 0; c < channel; ++c) {
  //   for(int h = 0; h < height; ++h) {
  //     for(int w = 0; w < width; ++w) {
  //       for(int k_h = 0; k_h < kernel_h; ++k_h) {
  //         for(int k_w = 0; k_w < kernel_w; ++k_w) { 
            
  //         }
  //       }
  //     }
  //   }
  // }



}

template 
void im2col<float>(const float* in, float* out);
template
void im2col<char>(const char* in, char* out);









}