# Configuration
$BUILD_DIR = "build"

Write-Host "Configuring project..." -ForegroundColor Cyan
# Using default generator (Visual Studio)
# ENABLE_GPU=ON to use GPU version of ONNX Runtime as specified in onnxruntime.cmake
cmake -B $BUILD_DIR -S . -DENABLE_GPU=ON

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
