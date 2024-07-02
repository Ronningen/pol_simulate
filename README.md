```shell
brew install libomp fftw eigen
pip install finufft
brew tap kazuakiyama/pgplot
brew install pgplot
brew tap kazuakiyama/difmap
brew install difmap
export PGPLOT_DIR=$PGPLOT_DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PGPLOT_DIR
export PGPLOT_DEV=/xwin
```

Download code for simulations of VLBI observations:
```shell
git clone --recursive https://github.com/Ronningen/pol_simulate.git
cd school2023
```

Make sure g++, eigen and boost are properly referenced in ``CMakeLists.txt``. E.g.:
```cmake
include_directories("/usr/include/eigen3", "include")
set(CMAKE_CXX_COMPILER /opt/homebrew/Cellar/gcc/14.1.0_1/bin/g++-14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lz -std=c++14 -fopenmp -march=native -DNDEBUG -O2 -fext-numeric-literals -I/opt/homebrew/Cellar/boost/1.85.0/include")
```

Compile C++ code for radiation transfer:
```shell
mkdir Release; cd Release
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

Run it:
```shell
python simulate.py --template_uvfits uvfits/1458+718.u.2006_09_06.uvf --redshift 0.4 --los_angle_deg 2. --cone_half_angle_deg 1.0 --Gamma 5.0 --B_1_Gauss 0.5 --m 1.0 --pitch_angle_deg 80 --tangled_fraction 0.0 --rot_angle_deg 30. --noise_scale_factor 0.3
```
First time it should fail because some Python libraries (e.g. ``astropy``, ``scikit-learn`` etc.) are not installed.
Install them and run again.

Command line options are described here:
```shell
python simulate.py -h
```
