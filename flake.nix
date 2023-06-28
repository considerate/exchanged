{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
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
    in
    {
      packages = forAllSystems (system: {
        default = pkgs.${system}.poetry2nix.mkPoetryApplication {
          python = pkgs.${system}.python310;
          projectDir = self;
          overrides = [ pkgs.${system}.poetry2nix.defaultPoetryOverrides python-overrides ];
        };
      });
      legacyPackages = pkgs;

      devShells = forAllSystems (system: {
        default = pkgs.${system}.mkShellNoCC {
          packages = with pkgs.${system}; [
            (poetry2nix.mkPoetryEnv {
              projectDir = self;
              python = pkgs.${system}.python310;
              overrides = [ poetry2nix.defaultPoetryOverrides python-overrides ];
            })
            poetry
          ];
        };
      });
    };
}
