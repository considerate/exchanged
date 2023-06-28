{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (system: import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      });
      PWD = builtins.getEnv "PWD";
      root = if PWD != "" then PWD else self;
      python-overrides = pyfinal: pyprev: {
        attrs = pyprev.attrs.overridePythonAttrs (old: {
          nativeBuildInputs = old.nativeBuildInputs ++ [
            pyfinal.hatchling
            pyfinal.hatch-fancy-pypi-readme
            pyfinal.hatch-vcs
          ];
        });
        maturin = pyprev.maturin.override { preferWheel = true; };
        ruff = pyprev.ruff.override { preferWheel = true; };
        ruff-lsp = pyprev.ruff.override { preferWheel = true; };
      };
      application = forAllSystems (system: pkgs.${system}.poetry2nix.mkPoetryApplication {
        projectDir = self;
        editablePackageSources = {
          exchanged = "${root}/src";
        };
        python = pkgs.${system}.python310;
        overrides = [ pkgs.${system}.poetry2nix.defaultPoetryOverrides python-overrides ];
        extraPackages = ps: [ ps.torch-bin ];
      });
      env = forAllSystems (system: pkgs.${system}.poetry2nix.mkPoetryEnv {
        projectDir = self;
        editablePackageSources = {
          exchanged = "${root}/src";
        };
        python = pkgs.${system}.python310;
        overrides = [ pkgs.${system}.poetry2nix.defaultPoetryOverrides python-overrides ];
        extraPackages = ps: [ ps.torch-bin ];
      });

    in
    {
      packages = forAllSystems (system: {
        default = application.${system};
      });
      legacyPackages = pkgs;

      devShells = forAllSystems (system: {
        default = pkgs.${system}.mkShellNoCC {
          packages = [
            env.${system}
            pkgs.${system}.python310.pkgs.ruff-lsp
            pkgs.${system}.pyright
            pkgs.${system}.poetry
          ];
        };
      });
    };
}
