include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

set(LIBFVAD_VERSION "1.0")
set(LIBFVAD_URL "https://github.com/dpirch/libfvad/archive/refs/tags/v${LIBFVAD_VERSION}.tar.gz")
set(LIBFVAD_SHA256 "8c8a1e4911a454b1a8e3ed062abbce4865eea3fc794417e2c96373309174215c")

FetchContent_Declare(
    libfvad
    URL ${LIBFVAD_URL}
    URL_HASH SHA256=${LIBFVAD_SHA256}
)

FetchContent_GetProperties(libfvad)
if(NOT libfvad_POPULATED)
    FetchContent_Populate(libfvad)
endif()

add_library(fvad STATIC
    ${libfvad_SOURCE_DIR}/src/fvad.c
    ${libfvad_SOURCE_DIR}/src/signal_processing/division_operations.c
    ${libfvad_SOURCE_DIR}/src/signal_processing/energy.c
    ${libfvad_SOURCE_DIR}/src/signal_processing/get_scaling_square.c
    ${libfvad_SOURCE_DIR}/src/signal_processing/resample_48khz.c
    ${libfvad_SOURCE_DIR}/src/signal_processing/resample_by_2_internal.c
    ${libfvad_SOURCE_DIR}/src/signal_processing/resample_fractional.c
    ${libfvad_SOURCE_DIR}/src/signal_processing/spl_inl.c
    ${libfvad_SOURCE_DIR}/src/vad/vad_core.c
    ${libfvad_SOURCE_DIR}/src/vad/vad_filterbank.c
    ${libfvad_SOURCE_DIR}/src/vad/vad_gmm.c
    ${libfvad_SOURCE_DIR}/src/vad/vad_sp.c
)
add_library(fvad::fvad ALIAS fvad)

set_target_properties(fvad PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS ON
)

target_include_directories(fvad
    PUBLIC
        ${libfvad_SOURCE_DIR}/include
    PRIVATE
        ${libfvad_SOURCE_DIR}/src
        ${libfvad_SOURCE_DIR}/src/vad
        ${libfvad_SOURCE_DIR}/src/signal_processing
)
