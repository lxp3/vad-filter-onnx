# Configuration
$VCVARS_PATH = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
$BUILD_DIR = "build"
$ONNX_FILE = "./downloads/onnxruntime-win-x64-gpu-1.17.1.zip"

Write-Host "--- Configuring vad-filter-onnx (VS 2026 Ninja) ---" -ForegroundColor Cyan

# if (Test-Path $BUILD_DIR) {
#     Write-Host "Cleaning existing build directory..."
#     Remove-Item -Path $BUILD_DIR -Recurse -Force
# }

# Run CMake Configuration inside a CMD environment with vcvarsall.bat
# Using Ninja for faster builds and cleaner output
# CMAKE_EXPORT_COMPILE_COMMANDS generates compile_commands.json for IntelliSense
$CmakeConfigCmd = "cmake -B $BUILD_DIR -S . -G `"Ninja`" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DONNXRUNTIME_FILE=$ONNX_FILE"

Write-Host "Executing CMake configuration..."
cmd.exe /c "`"$VCVARS_PATH`" x64 && $CmakeConfigCmd"

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "--- Building vad-filter-onnx ---" -ForegroundColor Cyan
$CmakeBuildCmd = "cmake --build $BUILD_DIR -j 16"
cmd.exe /c "`"$VCVARS_PATH`" x64 && $CmakeBuildCmd"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Build completed successfully!" -ForegroundColor Green
