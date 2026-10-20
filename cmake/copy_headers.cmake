# Copies *.h from SRC_DIR to DST_DIR (used by the release/ post-build step)
file(GLOB _hdrs "${SRC_DIR}/*.h")
file(COPY ${_hdrs} DESTINATION "${DST_DIR}")
