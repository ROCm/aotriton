# Use `table_tool.py` to update the schema of tuning database

Due to the limit of SQLite3, it is not possible to `ALTER TABLE <table> DROP CONSTRAINT`.
Therefore, the only way to update the constraint is create a new table and load the old data into the new table.

# Method 1: Create New Database file

Example usage:

## 1. Dump table to csv file

```
python -m v2python.table_tool --action dumpcsv \
		-f v2python/rules/tuning_database.sqlite3 \
		--table_name 'FLASH$attn_fwd' --table_file attn_fwd.csv
```

## 2. Create a new database with updated table

```
python tritonsrc/tune_flash.py \
		--db_file v2python/rules/new_tuning_database.sqlite3 \
		--stop_at 0 --create_table_only
```

## 3. Load dumped csv file to new database

```
python -m v2python.table_tool --action loadcsv \
	-f v2python/rules/new_tuning_database.sqlite3 \
	--table_name 'FLASH$attn_fwd' --table_file attn_fwd.csv -k ''
```

## 4. Fill the default values for new column(s)

```
sqlite3 v2python/rules/new_tuning_database.sqlite3 'update FLASH$attn_fwd set inputs$BIAS_TYPE = 0;'
```

## 5. Overwrite the old database

```
cp v2python/rules/new_tuning_database.sqlite3 v2python/rules/tuning_database.sqlite3
```

# Method 2: In-place update is also possible with the following steps:

1. Dump the table to csv file
2. Drop tables with sqlite3
3. Re-create tables in the same database file with `tune_flash.py --create_table_only`
4. Load dumped csv file to newly created tables
5. Fill the default values for the new column(s)
