cmake_minimum_required(VERSION 3.14)
project(ex LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "mpic++")

# 查找依赖包
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)
find_package(MPI REQUIRED)

# 设置不同平台的特定配置
if(APPLE)
    # macOS 上的 OpenMP 设置
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")

    # OpenBLAS 路径
    include_directories(/opt/homebrew/opt/openblas/include)
    link_directories(/opt/homebrew/opt/openblas/lib)

elseif(UNIX AND NOT APPLE)
    # Linux 上的 OpenMP 和 OpenBLAS 配置
    find_package(OpenBLAS REQUIRED)

    # 如果使用 OpenBLAS，确保链接库路径正确
    include_directories(${OpenBLAS_INCLUDE_DIR})
    link_directories(${OpenBLAS_LIBRARIES})
endif()

# 定义主要可执行文件的源文件列表
set(MAIN_SOURCES
    integral.cpp
)

# 遍历源文件列表，为每个源文件创建一个可执行目标
foreach(SOURCE ${MAIN_SOURCES})
    # 从文件名中提取目标名称（去除扩展名）
    get_filename_component(TARGET_NAME ${SOURCE} NAME_WE)
    
    # 为每个源文件创建可执行目标
    add_executable(${TARGET_NAME}
        ${SOURCE}
    )
    
    # 链接所需的库
    target_include_directories(${TARGET_NAME} PRIVATE
        ${EIGEN3_INCLUDE_DIRS}
        ${MPI_INCLUDE_DIRS}  # 使用正确的 MPI 头文件路径变量
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
        fmt::fmt
        OpenMP::OpenMP_CXX
        ${OpenBLAS_LIBRARIES}  # 使用 OpenBLAS 库
        Eigen3::Eigen
        ${MPI_LIBRARIES}      # 使用正确的 MPI 库路径变量
    )

    # 设置编译选项
    target_compile_options(${TARGET_NAME} PRIVATE 
        -Wall               # 启用所有警告
        -Wextra            # 启用额外警告
        -Wpedantic         # 启用严格的警告
        -O2                # 优化级别 2（提高性能）
        -g                 # 启用调试信息（如果需要调试）
    )
endforeach()
