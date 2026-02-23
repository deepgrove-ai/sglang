{
  pkgs,
  lib,
  config,
  inputs,
  ...
}
: {
  env.UV = "1";
  packages = with pkgs; [
  ];

  git-hooks.hooks = {
    shellcheck.enable = true;
    ruff.enable = true;
    ruff-format.enable = true;
    alejandra.enable = true;
  };

  languages.python = {
    package = pkgs.python312;
    libraries = [
      # Otherwise numpy yells at us for not being able find zlib.so
      pkgs.zlib
      # ImportError: libnuma.so.1: cannot open shared object file: No such file or directory
      # ModuleNotFoundError: No module named 'common_ops'
      pkgs.numactl
    ];
    enable = true;
    uv = {
      enable = true;
    };
    venv.enable = true;
  };
}
