Install ``finufft`` (for fast FT of the model images) and its Python bindings (``pyfinufft``):


```shell
git clone https://github.com/flatironinstitute/finufft.git
cd finufft
git checkout a8520208c
make test
make
pip install . --user
```


Download code for simulations of VLBI observations:
```shell
git clone --recursive https://github.com/ipashchenko/school2023.git
cd school2023
```

Install ``eigen`` linear algebra library. Make sure it is properly referenced in ``CMakeLists.txt``. E.g.:
```cmake
include_directories("/usr/include/eigen3", "include")
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
