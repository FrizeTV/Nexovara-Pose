#!/usr/bin/env bash
# Build Nexovara Pose SteamVR driver on Linux.
set -Eeuo pipefail
cd -- "$(dirname -- "$0")"

progress() {
  local pct="$1" label="$2" width=30 filled empty
  filled=$((pct * width / 100)); empty=$((width - filled))
  printf '\r[%.*s%*s] %3d%%  %s' "$filled" '##############################' "$empty" '' "$pct" "$label"
  [[ "$pct" == 100 ]] && printf '\n'
}

progress 0 "checking tools"
command -v curl >/dev/null 2>&1 || { echo; echo "Error: curl is required for portable downloads." >&2; exit 1; }

CMAKE_EXE=""
if command -v cmake >/dev/null 2>&1; then CMAKE_EXE="$(command -v cmake)"; fi
if [[ -z "$CMAKE_EXE" && -x "$PWD/tools/cmake/bin/cmake" ]]; then CMAKE_EXE="$PWD/tools/cmake/bin/cmake"; fi

if [[ -z "$CMAKE_EXE" ]]; then
  progress 10 "downloading portable CMake"
  mkdir -p tools
  cmake_version="3.31.6"
  archive="tools/cmake-portable.tar.gz"
  url="https://github.com/Kitware/CMake/releases/download/v${cmake_version}/cmake-${cmake_version}-linux-x86_64.tar.gz"
  curl -fL --progress-bar "$url" -o "$archive"
  rm -rf tools/cmake tools/cmake-extract
  mkdir -p tools/cmake-extract tools/cmake
  tar -xzf "$archive" -C tools/cmake-extract
  extracted="$(find tools/cmake-extract -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  [[ -n "$extracted" ]] || { echo; echo "Error: CMake archive is invalid." >&2; exit 1; }
  cp -a "$extracted"/. tools/cmake/
  rm -rf tools/cmake-extract "$archive"
  CMAKE_EXE="$PWD/tools/cmake/bin/cmake"
fi

progress 25 "configuring CMake"
command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || { echo; echo "Error: g++, clang++ or c++ is required." >&2; exit 1; }
"$CMAKE_EXE" -S driver -B driver/build -DCMAKE_BUILD_TYPE=Release -DNEXOVARAPOSE_BUILD_TESTS=ON -DNEXOVARAPOSE_DIST="$PWD/driver/dist"

progress 55 "compiling driver"
"$CMAKE_EXE" --build driver/build --config Release --parallel "${NEXOVARAPOSE_BUILD_JOBS:-$(nproc 2>/dev/null || echo 2)}"

progress 85 "running driver tests"
ctest --test-dir driver/build --build-config Release --output-on-failure
progress 100 "completed: driver/dist/nexovarapose"
