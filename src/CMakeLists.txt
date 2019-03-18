aux_source_directory(. DIR_LIB_SRC)
add_subdirectory(layer)
add_subdirectory(tools)
add_library(SrcFunctions ${DIR_LIB_SRC})
target_link_libraries(SrcFunctions ToolsFunctions LayerFunctions)

