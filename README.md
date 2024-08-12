# TensorRT Engine Inspector (layers)

#### Change this line for your engine

```cpp
// Load the TensorRT engine file
ifstream engineFile("../engine.trt", ios::binary);
```

#### How to launch?

```shell
# download repository
git clone https://github.com/Egorundel/tensorrt-engine_layers_inspector_cpp.git

# go to downloaded repository
cd tensorrt-engine_layers_inspector_cpp

# create `build` folder and go to her
mkdir build && cd build

# cmake 
cmake ..

# build it
cmake --build .
# or
make -j$(nproc)

# launch
./tensorrt-engine_layers_inspector_cpp
```

#### Screenshot of work

![screen0](./images/screen0.png)
