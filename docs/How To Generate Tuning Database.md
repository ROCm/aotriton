# TL;DR

``` bash
mkdir cpptune_build
cd cpptune_build
# -DCMAKE_INSTALL_PREFIX is mandatory to avoid conflicts with aotriton bundled by AOTriton
cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_BUILD_FOR_TUNING=ON -DAOTRITON_NAME_SUFFIX=123 -G Ninja
# Optionally only build for one arch
# cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_BUILD_FOR_TUNING=ON -DAOTRITON_OVERRIDE_TARGET_GPUS=gfx942_mod0 -DAOTRITON_NAME_SUFFIX=123 -G Ninja
ninja install
cd ..
# Run profiling on Target GPU
# We do not recommend updating the tuning_database.sqlite3 directly
PYTHONPATH=cpptune_build/bindings/ python test/tune_flash.py --json_file ~/navi32-aotriton_0.8.json --use_multigpu -1
# Update tuning database from experiment data stored in JSON file
python v2python/table_tool.py --action rawjson -k FLASH -f v2python/rules/tuning_database.sqlite3 -i ~/navi32-aotriton_0.8.json
```
