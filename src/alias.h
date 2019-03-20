#ifndef ALIAS_H_
#define ALIAS_H_

#include "./tensor.h"

#include <utility>
#include <vector>

namespace tinycnn {
  typedef std::vector<Tensor*> OUTPUTDATA;
  typedef std::vector<OUTPUTDATA> ALLDATA;

	typedef std::pair<int, std::vector<int>> INDEX;
	typedef std::vector<INDEX> INPUTINDEX;

  typedef std::vector<int> SHAPE;
}

#endif