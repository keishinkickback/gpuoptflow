package = "gpuoptflow"
version = "scm-1"

source = {
   url = "git://github.com/qassemoquab/gpuoptflow.git",
}

description = {
   summary = "Brox OpenCV optical flow on the GPU for torch",
   detailed = [[
   ]],
   homepage = "https://github.com/qassemoquab/gpuoptflow",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
   "cutorch >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
