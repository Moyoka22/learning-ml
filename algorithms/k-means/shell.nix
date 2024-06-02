{ pkgs ? import <nixpkgs> {} }:
with pkgs;
let 
  PROJECT_ROOT = builtins.getEnv "PWD";
  pythonPackages = ps: with ps; [ numpy pytest ipython ruff autopep8 ];
  py = python311.withPackages pythonPackages;
in
  mkShell {
    buildInputs = [ py ];
    shellHook = ''
    export fish_prompt_prefix='[problems]'
    exec fish
    '';
  }

