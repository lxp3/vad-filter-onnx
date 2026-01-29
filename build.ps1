# Configuration
# Configuration
$BUILD_SHARED_LIBS = "ON"
if ($BUILD_SHARED_LIBS -eq "ON") {
    $BUILD_DIR = "build_shared"
} else {
    $BUILD_DIR = "build_static"
}
Write-Host "Configuring project..." -ForegroundColor Cyan
# Using default generator (Visual Studio)
# ENABLE_GPU=ON to use GPU version of ONNX Runtime as specified in onnxruntime.cmake
cmake -B $BUILD_DIR -S . -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS -DENABLE_GPU=OFF -DENABLE_PYTHON=ON

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed!"
}

Write-Host "`n--- Building vad-filter-onnx ---" -ForegroundColor Cyan

# Build the project
cmake --build $BUILD_DIR --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Build completed successfully!" -ForegroundColor Green
