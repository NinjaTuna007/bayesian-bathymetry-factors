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
        # scikit-sparse (CHOLMOD) is not in nixpkgs; build it against nixpkgs' SuiteSparse.
        scikit-sparse = ps: ps.buildPythonPackage rec {
          pname = "scikit-sparse";
          version = "0.5.1";
          pyproject = true;
          src = ps.fetchPypi {
            pname = "scikit_sparse";
            inherit version;
            hash = "sha256-St7OPzeKyRhdF7qTSC6hm6h+VbEIjThfDCS8qJhK/Ag=";
          };
          build-system = with ps; [ setuptools wheel cython numpy ];
          buildInputs = [ pkgs.suitesparse ];
          dependencies = with ps; [ numpy scipy packaging ];
          env = {
            SUITESPARSE_INCLUDE_DIR = "${pkgs.lib.getDev pkgs.suitesparse}/include/suitesparse";
            SUITESPARSE_LIBRARY_DIR = "${pkgs.lib.getLib pkgs.suitesparse}/lib";
          };
          doCheck = false;
          pythonImportsCheck = [ "sksparse.cholmod" ];
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
          # sparse Cholesky for the Gauss-Newton steps
          (scikit-sparse ps)
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
