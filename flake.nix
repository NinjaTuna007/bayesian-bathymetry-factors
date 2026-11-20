{
  description = "SSS bathymetry: Python env for PingDSP 3DSS (pingdspinger) Örebro data";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # allowUnfree for the CUDA libraries behind jax-cuda13-plugin. cudaSupport stays off
        # globally so opencv & co. aren't rebuilt with CUDA; only JAX gets the GPU plugin.
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          # Quadro RTX 6000 (Turing). Without this, NCCL (needed by the JAX GPU plugin) is
          # compiled from source for every GPU architecture, which takes hours.
          config.cudaCapabilities = [ "7.5" ];
        };
        python = pkgs.python3.withPackages (ps: with ps; [
          # core
          numpy
          scipy
          pandas
          matplotlib
          tqdm
          # pingdspinger rosbag2 (.mcap) decoding without ROS; schemas are embedded in the bags
          rosbags
          # geo / point clouds
          pyproj
          rasterio
          xyzservices
          laspy
          # sss package: autodiff over the depths, GPU via the CUDA 13 PJRT plugin
          jax
          jaxlib
          jax-cuda13-plugin
          optax
          # notebooks
          ipykernel
          jupyterlab
          # tests
          pytest
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ python pkgs.ruff pkgs.pyright ];
        };
      });
}
