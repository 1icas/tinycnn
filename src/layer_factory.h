#ifndef LAYER_FACTORY_H_
#define LAYER_FACTORY_H_

#include <map>

#include "./error_handler.h"
#include "./layer.h"
#include "./macro.h"
#include "./param.h"

namespace tinycnn {

class Registry {
public:
	//for layers
	typedef Layer* (*Creator) (LayerParams* params);
	typedef std::map<int, Creator> LAYER_CONTAINER;
	//for layer params
	typedef LayerParams* (*ParamsCreator) ();
	typedef std::map<int, ParamsCreator> PARAMS_CONTAINER;

	static Registry& Get() {
		static Registry registry;
		return registry;
	}
	
	void insert(const int index, Creator creator, ParamsCreator pcreator) {
		insert_layer(index, creator);
		insert_params(index, pcreator);
	}

	void insert_layer(const int index, Creator creator) {
		if(layer_container_.find(index) != layer_container_.end())
			return;
		layer_container_[index] = creator;
	}

	Layer* create_layer(const int index, LayerParams* params) {
		if(layer_container_.find(index) != layer_container_.end())
			NOT_SUPPORT;
		return layer_container_[index](params);
	}

	void insert_params(const int index, ParamsCreator creator) {
		if(params_container_.find(index) != params_container_.end()) {
			return;
		}
		params_container_[index] = creator;
	}

	LayerParams* create_params(const int index) {
		if(params_container_.find(index) == params_container_.end()) {
			NOT_SUPPORT;
		}
		return params_container_[index]();
	}
private:
	LAYER_CONTAINER layer_container_;
	PARAMS_CONTAINER params_container_;
};	

class Creator {
public:
	Creator(const int index, Registry::Creator c, Registry::ParamsCreator p) {
		auto rgr = Registry::Get();
		rgr.insert(index, c, p);
		//rgr.insert_layer(index, c);
		//rgr.insert_params(index, p);
	}
};

#define Create_Layer_Func(name) \
	LayerParams* create_##name##_params() { \
		return new name##Params(); \
	} \
	Layer* create_##name##_layer(LayerParams* params) { \
		return new name##Layer(params); \
	}

#define Register(index, name) \
	Create_Layer_Func(name) \
	Creator name##_creator(index, create_##name##_layer, create_##name##_params);

}




#endif
