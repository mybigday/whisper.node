# AGENTS.md

Notes for working on whisper.node. Usage and the API are in [README.md](README.md).

## Release Builds

`.github/workflows/build-release.yml` builds every package (`release.yml` calls it on a tag, then publishes the root package). Run it by hand with `gh workflow run build-release.yml --ref <branch> -f target=win32-x64-cuda` to build one package (`target` is a glob over the package name without `node-whisper-`) without publishing; add `-f upload-artifacts=YES` to keep the packages as `bin-*` artifacts.

- `build`: one job per package, running `scripts/prepare-*` and `scripts/build-*` (and `scripts/build-wasm.sh` for `node-whisper-wasm`)
- `cuda-objects-*` / `build-cuda-*`: the CUDA packages. ggml-cuda is ~150 nvcc translation units, each compiled once per GPU architecture (about one core-hour per architecture), so `cuda-objects-*` jobs compile one slice of them each (`build-linux.sh --cuda-shard 2/3`, `build-windows.ps1 -cudaShard 2/8`) and upload the objects, and `build-cuda-*` build the rest and link the package from the objects of every slice (`--cuda-objects <dir>` / `-cudaObjects <dir>`). See `cmake/cuda-shards.cmake`. The number of slices is the `cuda-shard` list of the job. Any slice count gives the same package; a variant moved back to the matrix of `build` is built on one runner
- `.github/actions/setup-cuda`: installs the CUDA toolkit components a Windows build needs from NVIDIA's redistributable archives, pinned with checksums in `redist-<version>.json` next to it. To change the CUDA version, add that file for the new release (from NVIDIA's `redistrib_<version>.json`) and set `cuda-version` in the workflow
- The `publish` job of `release.yml` does not build anything native: it applies `scripts/whisper.cpp.patch` to the vendored sources (as a CMake configure would) and publishes the root package
- No compiler cache is used (`CCACHE_DISABLE=1`): tag runs cannot share caches with other runs, and the CUDA kernels change with every whisper.cpp bump
