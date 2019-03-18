#include <vector>

#include "../alias.h"
#include "../error_handler.h"

namespace tinycnn {

void _pad(const SHAPE& shape, 
          const int kernel_w, 
          const int kernel_h,
          const int stride_w,
          const int stride_h,
          int& pad_along_width,
          int& pad_along_height,
          int& pad_top,
          int& pad_bottom,
          int& pad_left,
          int& pad_right
        ) {
  EQUAL(shape.size(), 4);
  pad_along_height = std::max(shape[1] % stride_h == 0 ? 
                            (kernel_h - stride_h) : kernel_h - (shape[1] % stride_h), 0);
  pad_along_width = std::max(shape[2] % stride_w == 0 ? 
                            (kernel_w - stride_w) : kernel_w - (shape[2] % stride_w), 0);
  pad_top = pad_along_height / 2;
  pad_bottom = pad_along_height - pad_top;
  pad_left = pad_along_width / 2;
  pad_right = pad_along_width - pad_left;
}

void _conv_output(const int width,
                  const int height,
                  const int k_w,
                  const int k_h,
                  const int pad_left,
                  const int pad_right,
                  const int pad_top,
                  const int pad_bottom,
                  const int stride_w,
                  const int stride_h,
                  int& output_w,
                  int& output_h) {
  auto func = [&](const int a, const int b, const int c, const int d, const int e) -> int {return (a - b + c + d) / e + 1;};
  output_w = func(width, k_w, pad_left, pad_right, stride_w);
  output_h = func(height, k_h, pad_top, pad_bottom, stride_h);
  }

}