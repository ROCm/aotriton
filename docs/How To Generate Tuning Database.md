# TL;DR

```
mkdir cpptune_build
cd cpptune_build
cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_BUILD_FOR_TUNING=ON -G Ninja
# Optionally only build for one arch
# cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_BUILD_FOR_TUNING=ON -DTARGET_GPUS=Navi32 -G Ninja
ninja install
cd ..
PYTHONPATH=cpptune_build/bindings/ python test/tune_flash.py --bias_type 0 --db_file v2python/rules/tuning_database.sqlite3
```
