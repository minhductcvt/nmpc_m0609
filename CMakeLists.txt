cmake_minimum_required(VERSION 3.8)
project(dsr_m0609_nmpc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Use DT_RPATH instead of DT_RUNPATH so transitive dependencies (libhpipm, libblasfeo) are found
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(dsr_msgs2 REQUIRED)

# Auto-detect acados root (-DACADOS_DIR=..., $ACADOS_SOURCE_DIR, $ACADOS_DIR, ~/acados, /opt/acados, <ws>/acados)
if(NOT DEFINED ACADOS_DIR OR NOT EXISTS "${ACADOS_DIR}/include")
  set(_ACADOS_CANDIDATES
    "$ENV{ACADOS_SOURCE_DIR}"
    "$ENV{ACADOS_DIR}"
    "$ENV{HOME}/acados"
    "/opt/acados"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../acados"
    "/usr/local/acados"
  )
  foreach(_cand ${_ACADOS_CANDIDATES})
    if(_cand AND EXISTS "${_cand}/include/acados")
      set(ACADOS_DIR "${_cand}")
      break()
    endif()
  endforeach()
endif()
if(NOT DEFINED ACADOS_DIR OR NOT EXISTS "${ACADOS_DIR}/include/acados")
  message(FATAL_ERROR
    "Could not locate acados! Set ACADOS_SOURCE_DIR env var, install at ~/acados or /opt/acados, or pass -DACADOS_DIR=/path/to/acados.")
endif()
message(STATUS "Using ACADOS_DIR: ${ACADOS_DIR}")
set(SOLVER_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_solver_kinematic")

include_directories(
  include
  ${ACADOS_DIR}/include
  ${ACADOS_DIR}/include/acados
  ${ACADOS_DIR}/include/blasfeo/include
  ${ACADOS_DIR}/include/hpipm/include
  c_generated_solver_kinematic
  c_generated_solver_kinematic/m0609_kinematic_nmpc_cost
)

link_directories(
  ${ACADOS_DIR}/lib
  ${SOLVER_LIB_DIR}
)

# C++ NMPC Node executable
add_executable(m0609_nmpc_node
  src/m0609_nmpc_node.cpp
)

ament_target_dependencies(m0609_nmpc_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  trajectory_msgs
  dsr_msgs2
)

target_link_libraries(m0609_nmpc_node
  ${SOLVER_LIB_DIR}/libacados_ocp_solver_ocp_m0609_kinematic_nmpc_41c0c9b9.so
  ${ACADOS_DIR}/lib/libacados.so
  ${ACADOS_DIR}/lib/libhpipm.so
  ${ACADOS_DIR}/lib/libblasfeo.so
  m
)

set_target_properties(m0609_nmpc_node PROPERTIES
  BUILD_RPATH "${ACADOS_DIR}/lib;${SOLVER_LIB_DIR}"
  INSTALL_RPATH "${ACADOS_DIR}/lib;${SOLVER_LIB_DIR};\$ORIGIN/..;\$ORIGIN"
)

# Install executable and launch files
install(TARGETS
  m0609_nmpc_node
  DESTINATION lib/${PROJECT_NAME}
)

file(GLOB ACADOS_LIB_FILES "${ACADOS_DIR}/lib/lib*.so*")
install(FILES
  ${SOLVER_LIB_DIR}/libacados_ocp_solver_ocp_m0609_kinematic_nmpc_41c0c9b9.so
  ${ACADOS_LIB_FILES}
  DESTINATION lib
)

install(PROGRAMS
  scripts/pick_and_place_mission.py
  scripts/dsr_teach_pendant_gui.py
  scripts/dsr_vision_node.py
  scripts/dynamic_obstacle_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  model
  worlds
  config
  programs
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
