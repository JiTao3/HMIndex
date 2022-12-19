# HMIndex

----------------------

## Required Libraries

- LibTorch: 
  - HomePage: https://pytorch.org/get-started/locally/
  - Download: https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.12.0%2Bcpu.zip
  - GPU Version: https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.12.0%2Bcu113.zip

- Eigen
  - HomePage: https://eigen.tuxfamily.org/index.php?title=Main_Page
  - Download: https://gitlab.com/libeigen/eigen/-/releases/3.4.0

## Use Libraries

You need to change the library path in `CMakeLists.txt` file like this, replace `$yourpath` with the correct library file path:

```bash
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} "$yourpath/nvlab/kittool")
set(Torch_DIR "$yourpath/libtorch_gpu/libtorch/share/cmake/Torch")
include_directories("$yourpath/eigen-3.4.0/")
```

## Train the Meta Optimizer

### Buile the DB-Tree and Save each leaf node.

```bash
cd build 
cmake .. 
make 
./build/DBTree $datasetpath/dataset/OSM/osm.csv $savepath/dataset/OSM/split2/
```

Modifying the Meta Optimizer Configuration File in `Psrc/`:

```yml
split_data: "$savepath/dataset/OSM/split2/"
```

### Train Meta Optimizer

Now you can train the Meta Optimizet and a meta optimizer model will be save in the file `$savepath`: 

```bash
python -u main.py -exp_name osm_ne_us_1 -save_path $savepath
```

### Setup Meta Optimizer

Then you can setup a Meta Optimizer server:

```bash
leoModelPath = "$savepath"
python -u Psrc/main.py
```
### Query Experiments

You can use the meta-optimizer to train the model while building HM-Index, or to batch generate initialization parameters and load the trained parameters.

You need to modify the following and compile the source file.

`rangeQueryPrefix` is used to load different range query window sizes and aspect ratios.

```cpp
// Point
    data_space_bound = {};
    csv_path = "";
    model_param_path = "";
    query_path = "";

// range
    data_space_bound = {};
    csv_path = "";
    model_param_path = "";
    rangeQueryPrefix = "";

// kNN
    data_space_bound = {};
    csv_path = "";
    model_param_path = "";
    queryPoints = knnQueryFileReader->get_array_points("", ",");

```

If you want to build and train the model at the same time, You can call `void getParamFromScoket(int serverPort, vector<MetaData> &metadataVec);` in the train function `void CellTree::train(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root)` of `CellTree.cpp`.

Here is an example of an experiment with a uniform distributed dataset and queries.

```bash
cd build 
cmake .. 
make 
./build/HM-Index uniform >> log/uniform_v1.log 
```


### Others 

Naming issues: CellTree is called HM-Index.



