#!/usr/bin/env python

import sqlite3
import itertools
from collections import defaultdict
import json
from copy import deepcopy
import argparse
import sys
import os
import math
import numpy as np
import csv
from tqdm import tqdm
from pathlib import Path

# FIXME: load from kdesc
HEAD_DIMS = np.array([16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512], dtype=np.int32)
SEQLENS = np.array([16,32,64,128,256,512,1024,2048,4096,8192], dtype=np.int32)
ROUND_INPUTS = bool(int(os.getenv('ROUND_INPUTS', True)))

def round_to_power_of_two(x):
    return 2 ** (x - 1).bit_length()

def round_to_array(x, arr):
    return int(np.min(arr[arr >= x]))

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', type=str, default=None, help='Input CSV/JSON file')
    p.add_argument('-f', '--file', type=str, default=None, help='Database file')
    p.add_argument('--opfile', type=Path, default=None, help='Op Database file')
    p.add_argument('-k', '--kernel_family', type=str, help='Kernel family')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    p.add_argument('--action', type=str, required=False, default='pipejson',
                   choices=['pipejson', 'createtableonly', 'dumpcsv', 'loadcsv', 'rawjson', 'rawjson_fudge_check', 'opjson', 'rawsc'],
                   help='Action to perform. pipejson means directly inserting json objects into the database. rawjson means aggregating the raw profiling log (generated from the raw cpptune json objects) and insert the best kernel to database.')
    p.add_argument('--table_name', type=str, help='Table to dump/load')
    p.add_argument('--table_file', type=str, help='CSV file of dump/load')
    p.add_argument('--select_where', type=str, default='', help='Extra WHERE clause for SQL to only dump selected rows to CSV file')
    p.add_argument('--ignore_id', action='store_true', help='Ignore row IDs when loading CSV to database, useful for table merge')
    p.add_argument('--round_inputs', action='store_true', help='Round seqlens and hdims of input json. Will fail if any inputs does not need rounding. If only hdim is rounded PADDED_HEAD=False entry will not be updated')
    p.add_argument('--fudge_factor_tolerance', type=float, default=5.0,
                   help='''For rawjson mode.
                   During the profiling, a "target" fudge factor is computed as
                   the minimal fudge factor that allows the kernel to pass the
                   unit tests.
                   However kernels compiled with different options have
                   different precisions, and the fatest kernel may have
                   horrible number errors.
                   This tolerance (called T) is a threshold factor to only
                   select fatest kernels within a subset of kernel whose target
                   fudge factor is smaller than T * "best target fudge factors".
                   ''')
    p.add_argument('--max_fudge_factor', type=float, default=100.0)
    p.add_argument('--sc_report', type=str, default=None, help='Write san check results to this JSON file. Required for --action rawsc')
    args = p.parse_args()
    if args.action == 'rawsc':
        assert args.sc_report is not None, '--sc_report is required for --action rawsc'
        assert args.sc_report.endswith('.sc_report'), "For safety, --sc_report file must use .sc_report suffix to avoid overwritting raw json file"
    if args.action != 'rawsc':
        assert args.kernel_family, f'--kernel_family is needed for --action {args.action}'
    return args

# TODO: Refactor this piece
#       Use --kernel_family to lookup info

class PerKernelResult(object):
    KERNEL_NAME = None
    KERNEL_OUT_TENSORS = []
    KERNEL_MAX_FUDGE_FACTORS = None

    def __init__(self, task_id):
        self._tid = task_id
        self._jarray = []

    @property
    def tid(self):
        return self._tid

    def collect(self, j):
        self._jarray.append(j)

    def conclude(self):
        self.valid_out_tensors = self.KERNEL_OUT_TENSORS

    def _get_most_accurate_kernel_tft(self):
        tfts = { tn : [] for tn in self.valid_out_tensors }
        for j in self._jarray:
            if 'target_fudge_factors' not in j:
                continue
            if j['target_fudge_factors'] is None:
                continue
            for tn in self.valid_out_tensors:
                tft = j['target_fudge_factors'].get(tn, None)
                if tft is not None:
                    tfts[tn].append(tft)
        for tft in tfts.values():
            if len(tft) == 0:
                return None
        # FIXME: It is possible that one kernel excels at one tensor, while another kernel at another,
        #        and at the end of the day no kernel meets the error tolerances
        #        of two tensors at the same time.
        #        Although unlikely for now but eventually we'll need a
        #        resolution for this (use sum of target_fudge_factors?)
        return { tn : max(1.0, np.min(tfts[tn])) for tn in self.valid_out_tensors }

    def _get_optimal_kernel_tft(self, fudge_factor_tolerance, max_fudge_factor, allow_no_acceptable=False):
        verbose = False
        best_tft = self._get_most_accurate_kernel_tft()
        if allow_no_acceptable and best_tft is None:
            return None
        if best_tft is None:
            print(f'NEED RERUN TID: {self.tid}')
            return None
        fft = fudge_factor_tolerance
        def is_acceptable(j):
            if j['result'] != 'tuned':
                if verbose:
                    print('is_acceptable: result != tuned')
                return False
            if not isinstance(j['time'], list):
                t = j['time']
                if not isinstance(t, float):
                    if verbose:
                        print('is_acceptable: time is not float')
                    return False
                if math.isnan(t) or math.isinf(t):
                    if verbose:
                        print('is_acceptable: time is nan/inf')
                    return False
            adiffs = j['adiffs']
            if adiffs is None or self.any_nan(adiffs):
                if verbose:
                    print(f'is_acceptable: {adiffs=}')
                return False
            fits = { tn : j['target_fudge_factors'][tn] < min(max_fudge_factor, fft * best_tft[tn]) for tn in self.valid_out_tensors }
            if verbose:
                print(f'{fits=}')
            return all(fits.values())
        acceptables = list(filter(is_acceptable, self._jarray))
        def gettime(j):
            if not isinstance(j['time'], list):
                return j['time']
            return tuple(j['time'])
        if not acceptables:
            if allow_no_acceptable:
                return None
            # print(f'{best_tft=}')
            assert False, 'acceptables is empty'
        # print(f'{acceptables=}')
        optimal = min(acceptables, key=gettime)
        optimal = self.remove_unused(optimal)
        return optimal

    def sensible_gen(self):
        for j in self._jarray:
            if j['result'] != 'tuned':
                continue
            if 'target_fudge_factors' not in j:
                continue
            if j['target_fudge_factors'] is None:
                continue
            if not isinstance(j['time'], list):
                t = j['time']
                if not isinstance(t, float):
                    continue
                if math.isnan(t) or math.isinf(t):
                    continue
            adiffs = j['adiffs']
            if adiffs is None or self.any_nan(adiffs):
                continue
            yield j

    def get_accurate_kernels(self, tolerance_factor):
        best_adiffs = None
        sensibles = list(self.sensible_gen())
        for j in sensibles:
            if best_adiffs is None:
                best_adiffs = j['adiffs']
            elif best_adiffs > j['adiffs']:
                best_adiffs = j['adiffs']
        if best_adiffs is None:
            return None, None
        if isinstance(best_adiffs, list):
            acceptable_adiffs = [ tolerance_factor * e for e in best_adiffs ]
        else:
            acceptable_adiffs = tolerance_factor * best_adiffs
        return best_adiffs, [ j for j in sensibles if j['adiffs'] < acceptable_adiffs ]

    def get_optimal_kernel(self, fudge_factor_tolerance, max_fudge_factor, allow_no_acceptable=False):
        verbose = False
        best_adiffs, acceptables = self.get_accurate_kernels(fudge_factor_tolerance)

        if allow_no_acceptable and best_adiffs is None:
            return None
        if best_adiffs is None:
            print(f'NEED RERUN TID: {self.tid}')
            return None

        def gettime(j):
            if not isinstance(j['time'], list):
                return j['time']
            return tuple(j['time'])
        if not acceptables:
            if allow_no_acceptable:
                return None
            # print(f'{best_tft=}')
            assert False, 'acceptables is empty'
        # print(f'{acceptables=}')
        # for acceptable in acceptables:
        #     print(f'{acceptable=}')
        optimal = min(acceptables, key=gettime)
        # print(f'{best_adiffs=} {optimal["adiffs"]=}')
        optimal = self.remove_unused(optimal)
        return optimal

    def any_nan(self, adiffs):
        if isinstance(adiffs, float):
            return math.isnan(adiffs)
        else:
            return any(map(math.isnan, adiffs))

    def remove_unused(self, optimal):
        return optimal

    @classmethod
    def update_max_fudge_factor(klass, opti):
        if opti is None:
            return
        if klass.KERNEL_MAX_FUDGE_FACTORS is None:
            klass.KERNEL_MAX_FUDGE_FACTORS = { tn : 0.0 for tn in klass.KERNEL_OUT_TENSORS }
        for tn in klass.KERNEL_MAX_FUDGE_FACTORS:
            otff = opti['target_fudge_factors'][tn]
            if otff is None:
                continue
            klass.KERNEL_MAX_FUDGE_FACTORS[tn] = max(klass.KERNEL_MAX_FUDGE_FACTORS[tn], otff)

class FAPkr(PerKernelResult):
    def entry_from_json(self):
        head = self._jarray[0]
        inputs = head['inputs']
        d = {'arch' : head['arch']}
        d['causal_type'] = inputs['CAUSAL_TYPE']
        d['d_head'] = inputs['BLOCK_DMODEL']
        d['dropout_p'] = 0.5 if inputs['ENABLE_DROPOUT'] else 0.0
        d['dtype'] = inputs['Q_dtype'].removeprefix('torch.')
        d['bias_type'] = inputs['BIAS_TYPE']
        d['seqlen_q'] = inputs['Max_seqlen_q']
        d['seqlen_k'] = inputs['Max_seqlen_k']
        return json.dumps(d)

class Pkr_AttnFwd(FAPkr):
    KERNEL_NAME = 'attn_fwd'
    KERNEL_OUT_TENSORS = ['out']

    def remove_unused(self, optimal):
        for key in ['BATCH', 'N_HEADS', 'D_HEAD', 'RETURN_ENCODED_SOFTMAX']:
            if key in optimal['inputs']:
                del optimal['inputs'][key]
        return optimal

class Pkr_BwdKernelDkDv(FAPkr):
    KERNEL_NAME = 'bwd_kernel_dk_dv'
    KERNEL_OUT_TENSORS = ['dk', 'dv']

    def remove_unused(self, optimal):
        for key in ['BATCH', 'N_HEADS', 'D_HEAD', 'RETURN_ENCODED_SOFTMAX'] + ['USE_ALIBI', 'INT8', 'INT8_KV', 'USE_P_SCALE']:
            if key in optimal['inputs']:
                del optimal['inputs'][key]
        return optimal

class Pkr_BwdKernelDq(FAPkr):
    KERNEL_NAME = 'bwd_kernel_dq'
    KERNEL_OUT_TENSORS = ['dq', 'db']

    def conclude(self):
        self.valid_out_tensors = self.KERNEL_OUT_TENSORS
        if self._jarray[0]['inputs']['BIAS_TYPE'] == 0:
            self.valid_out_tensors = ['dq']

    def any_nan(self, adiffs):
        # TODO: shouldn't this be == 1?
        if len(self.valid_out_tensors) == 0:  # db isn't there
            return math.isnan(adiffs[0])
        return any(map(math.isnan, adiffs))

    remove_unused = Pkr_BwdKernelDkDv.remove_unused

class Pkr_FusedBwdKernel(FAPkr):
    KERNEL_NAME = 'bwd_kernel_fuse'
    KERNEL_OUT_TENSORS = ['dk', 'dv', 'dq', 'db']
    KERNEL_OUT_TENSORS_NOBIAS = ['dk', 'dv', 'dq', 'db']

    def conclude(self):
        bias = self._jarray[0]['inputs']['BIAS_TYPE']
        self.valid_out_tensors = self.KERNEL_OUT_TENSORS if bias else self.KERNEL_OUT_TENSORS_NOBIAS

    def any_nan(self, adiffs):
        ntensors = len(self.valid_out_tensors)
        return any(map(math.isnan, adiffs[:ntensors]))

    remove_unused = Pkr_BwdKernelDkDv.remove_unused

class Pkr_OpAttnBwd(Pkr_FusedBwdKernel):
    KERNEL_NAME = 'op_attn_bwd'

KERNEL_NAME_TO_FACTORY = {
    'attn_fwd' : Pkr_AttnFwd,
    'bwd_kernel_dk_dv' : Pkr_BwdKernelDkDv,
    'bwd_kernel_dq' : Pkr_BwdKernelDq,
    'bwd_kernel_fuse' : Pkr_FusedBwdKernel,
    'op_attn_bwd' : Pkr_OpAttnBwd,
}

def pkr_factory(key):
    arch, tid, kn = key
    factory = KERNEL_NAME_TO_FACTORY[kn]
    return factory(tid)

class TuningDatabase(object):
    OPTABLE = False
    PYTYPE_TO_SQLTYPE = {
        int : 'INTEGER',
        str : 'TEXT',
        float : 'REAL',  # Not used right now. Possible to store actual profiling results
    }

    def __init__(self, args):
        self._args = args
        if args.file is not None:
            self._conn = sqlite3.connect(args.file)  # TODO: use autocommit for python 3.12+
            self._conn.isolation_level = None  # TODO: add --batch mode,
            if args.opfile is not None:
                self._conn.execute(f"ATTACH DATABASE '{args.opfile.as_posix()}' AS op;")
            self._cur = self._conn.cursor()
        else:
            self._conn = None
        self._table_existance_checked = set()

    @property
    def verbose(self):
        return self._args.verbose

    def value_to_pytype(self, v):
        if isinstance(v, int):
            return int
        if isinstance(v, str):
            return str
        if isinstance(v, float):
            return float
        assert False, f"Unsupported type for value {v} {v.__class__=}"

    def sqltype(self, pytype):
        return self.PYTYPE_TO_SQLTYPE[pytype]

    def collect_columns(self, sub_tune_info: dict, *, prefix='', sans=()):
        ret = []
        for k, v in sub_tune_info.items():
            # Do not store Q.shape and others, which is an array and makes things more complicated
            if k.endswith('.shape'):
                continue
            if k in sans:
                continue
            tup = (f'{prefix}{k}', v, self.value_to_pytype(v))
            ret.append(tup)
        if self.verbose:
            print(f'collect_columns: {ret=}')
        return ret

    def get_table_name(self, tune_info: dict) -> str:
        kn = tune_info['kernel_name']
        return f'{self._args.kernel_family}${kn}'

    def ensure_table(self, tune_info : dict) -> str:
        kn = tune_info['kernel_name']
        if kn in self._table_existance_checked:
            return self.get_table_name(tune_info)
        table_name = self._create_table(tune_info)
        self._table_existance_checked.add(kn)
        return table_name

    def _create_table(self, tune_info):
        columns = self.collect_columns(tune_info['inputs'], prefix='inputs$')
        # UNIQUE = 'UNIQUE'
        col_def = ['id INTEGER PRIMARY KEY', f'gpu TEXT']
        col_def += [f'{colname} {self.sqltype(pytype)}' for colname, _, pytype in columns]
        unique = ', '.join(['gpu'] + [colname for colname, _, _ in columns])
        columns = self.collect_columns(tune_info['tuned_kernel'], prefix='tuned_kernel$')
        col_def += [f'{colname} {self.sqltype(pytype)}' for colname, _, pytype in columns]
        columns = self.collect_columns(tune_info['compiler_options'], prefix='compiler_options$')
        col_def += [f'{colname} {self.sqltype(pytype)}' for colname, _, pytype in columns]
        col_def_stmt = ', '.join(col_def)
        table_name = self.get_table_name(tune_info)
        stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({col_def_stmt}, UNIQUE({unique}));"
        if self.verbose:
            print("Executing", stmt)
        self._cur.execute(stmt)
        self._conn.commit()
        return table_name

    def upsert(self, line_text, *, create_table_only):
        if not line_text:
            return
        tune_info = json.loads(line_text)
        if self.verbose:
            print(f'{line_text=}')
            print(f'{tune_info=}')
        self.upsert_json(tune_info, create_table_only=create_table_only)

    def upsert_json(self, tune_info, *, create_table_only):
        tune_result = tune_info.get('result', 'result-not-reported-in-older-version')
        if not tune_result == 'tuned':
            if self.verbose:
                print(f'{tune_result=}')
            return
        sql_table = self.ensure_table(tune_info)
        if create_table_only:
            return
        inputs_columns = self.collect_columns(tune_info['inputs'], prefix='inputs$', sans=('BATCH'))
        if self.OPTABLE:
            tuned_kernel_columns = self.collect_columns(tune_info['op'], prefix='op$')
            compiler_options_columns = []
        else:
            tuned_kernel_columns = self.collect_columns(tune_info['tuned_kernel'], prefix='tuned_kernel$')
            compiler_options_columns = self.collect_columns(tune_info['compiler_options'], prefix='compiler_options$')
        all_colnames = ['gpu'] + [colname for colname, _, _ in itertools.chain(inputs_columns, tuned_kernel_columns, compiler_options_columns)]
        stmt_colnames = ', '.join(all_colnames)
        stmt_placeholders = ', '.join(['?'] * len(all_colnames))
        stmt = f'INSERT INTO {sql_table}({stmt_colnames}) VALUES({stmt_placeholders})'
        gpu = tune_info['arch'] + '_mod0'  # FIXME: non-mod0 gpu
        values = [gpu] + [v for _, v, _ in itertools.chain(inputs_columns, tuned_kernel_columns, compiler_options_columns)]
        if self.verbose:
            print("values 1: ", values)
        stmt += ' ON CONFLICT DO UPDATE SET '
        stmt += ', '.join([f'{colname}=?' for colname, _, _ in itertools.chain(tuned_kernel_columns, compiler_options_columns)])
        values += [v for _, v, _ in itertools.chain(tuned_kernel_columns, compiler_options_columns)]
        if self.verbose:
            print("values 2: ", values)
        if self.verbose:
            print("Executing", stmt, "with", values)
        self._cur.execute(stmt, values)
        self._conn.commit()

    def close(self):
        if self._args.file is not None:
            self._cur.close()
            self._conn.close()

    def dumpcsv(self, table_name, table_file):
        with open(table_file, mode='w', newline='') as file:
            stmt = f"SELECT * FROM {table_name}"
            if self._args.select_where:
                stmt += f" WHERE {self._args.select_where}"
            stmt += ";"
            self._cur.execute(stmt)
            writer = csv.writer(file)
            colunm_names = [tup[0] for tup in self._cur.description]
            writer.writerow(colunm_names)
            while True:
                tup = self._cur.fetchone()
                if tup is None:
                    break
                writer.writerow(tup)

    def loadcsv(self, table_file, table_name):
        pbar = tqdm(desc='Processed lines')
        with open(table_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            csv_headers = next(reader)
            pbar.update(1)
            if self._args.ignore_id:
                assert csv_headers[0] == 'id', "--ignore_id: First column of CSV is not 'id'. This tool does not handle more compilicated situations."
                csv_headers = csv_headers[1:]
            colunm_names = ', '.join(csv_headers)
            placeholders = ', '.join(['?'] * len(csv_headers))
            stmt = f'INSERT INTO {table_name} ({colunm_names}) VALUES({placeholders})'
            tuned_kernel_columns = [ cname for cname in csv_headers if cname.startswith('tuned_kernel$') ]
            compiler_options_columns = [ cname for cname in csv_headers if cname.startswith('compiler_options$') ]
            both_columns = tuned_kernel_columns + compiler_options_columns
            cindices = []
            for i, cname in enumerate(csv_headers):
                if cname in both_columns:
                    cindices.append(i)
            stmt += ' ON CONFLICT DO UPDATE SET '
            stmt += ', '.join([f'{colname}=?' for colname in both_columns])
            print('stmt', stmt)
            print(f'{cindices=} {len(cindices)=} {len(csv_headers)=}')
            for i in cindices:
                print(f'{i=}: {csv_headers[i]=}')
            for row in reader:
                pbar.update(1)
                if self._args.ignore_id:
                    row = row[1:]
                for i in cindices:
                    row.append(row[i])
                self._cur.execute(stmt, row)
            self._conn.commit()

    def init_aggregation(self):
        self.pkr_database = {}  # dict: (gpu, task_id, kernel_name) -> (best, json)

    def aggregate(self, line_text):
        round_inputs = self._args.round_inputs
        if not line_text:
            return
        raw_info = json.loads(line_text)
        if raw_info.get("result", None) == "skipped":
            return
        kernel_name = raw_info.get('kernel_name', '')
        if raw_info.get('kernel_name') == 'attn_fwd':
            BM = raw_info['tuned_kernel']['BLOCK_M']
            BN = raw_info['tuned_kernel']['BLOCK_N']
            if BM < BN:
                # print(raw_info)
                # Known faulty kernel for fp32
                return
            # PRE_LOAD_V = raw_info['tuned_kernel']['PRE_LOAD_V']
            # # Skip PRE_LOAD_V=2
            # if PRE_LOAD_V:
            #     return
        if raw_info.get('tuned_kernel', None) is not None:
            # Skip num_stages=2
            if raw_info['compiler_options']['num_stages'] == 2:
                return
        if not raw_info['inputs']['Q_dtype'].startswith('torch.'):
            raw_info['inputs']['Q_dtype'] = 'torch.' + raw_info['inputs']['Q_dtype']
        # Workaround to fix a bug where BLOCK_DMODEL was not correctly rounded
        # in mptune/flash/db_accessor.py
        raw_info['inputs']['BLOCK_DMODEL'] = round_to_array(raw_info['inputs']['D_HEAD'], HEAD_DIMS)
        if raw_info['inputs']['BLOCK_DMODEL'] == raw_info['inputs']['D_HEAD']:
            raw_info['inputs']['PADDED_HEAD'] = False
        raw_info['inputs']['CAUSAL_TYPE'] = 3 if raw_info['inputs']['CAUSAL_TYPE'] == 1 else 0

        # Workaround to fix large block size for hdim=512 on navi31
        # if raw_info['arch'] in ['gfx1100', 'gfx1201'] and raw_info['inputs']['BLOCK_DMODEL'] == 512:
        #     if 'tuned_kernel' in raw_info:
        #         ti = raw_info['tuned_kernel']
        #         # print(f"{ti=}")
        #         if ti['BLOCK_M'] > 16 or ti['BLOCK_N'] > 16:
        #             return

        # For OpTune: move 'inputs'/'op_backend' and 'tflops' to 'op': {...}
        # FIXME: Correctly store info with Tuner V3
        if 'op_backend' in raw_info['inputs']:
            op = {
                'backend' : raw_info['inputs']['op_backend'],
                'tflops' : raw_info.get('TFLOPS', float("NaN")),
            }
            raw_info['op'] = op
            del raw_info['inputs']['op_backend']
            raw_info['kernel_name'] = 'op_attn_bwd'

        def rounding(check_only):
            need_rounding_keys = []
            # Round D_HEAD
            # It is not used, but it is part of UNIQUE constraints
            def round_hdims(x):
                return round_to_array(x, HEAD_DIMS)
            def round_seqlen(x):
                return round_to_array(x, SEQLENS)
            for key, rf in [ ('D_HEAD', round_hdims),
                             ('Max_seqlen_q', round_seqlen),
                             ('Max_seqlen_k', round_seqlen),
                           ]:
                if key in raw_info['inputs']:
                    oldv = raw_info['inputs'][key]
                    newv = rf(oldv)
                    if oldv == newv:
                        continue
                    if check_only:
                        # print(f'{key=} {oldv=} {newv=}')
                        need_rounding_keys.append(key)
                    else:
                        raw_info['inputs'][key] = newv
            return need_rounding_keys
        need_rounding_keys = rounding(check_only=True)
        need_rounding = len(need_rounding_keys) > 0
        # if need_rounding != round_inputs:
        #     print(raw_info)

        # assert need_rounding == round_inputs, '--round_inputs should only be applied to json with irregular inputs, and vise versa'
        # if len(need_rounding_keys) == 1 and 'D_HEAD' in need_rounding_keys:
        #     only_hdim_rounded = True
        # else:
        only_hdim_rounded = False
        if round_inputs:
            rounding(check_only=False)
            if not only_hdim_rounded:
                raw_info['inputs']['PADDED_HEAD'] = False
            else:
                assert raw_info['inputs']['PADDED_HEAD'] == True, f'{need_rounding_keys=}'
        else:
            assert raw_info['inputs']['PADDED_HEAD'] == False
        timing = raw_info.get('time', float('inf'))
        if isinstance(timing, float):
            if math.isinf(timing):
                return
            if not self.OPTABLE:
                assert False, f'time element in raw json log must be a list or float("inf") but get {timing}'
        if self.OPTABLE:
            # FIXME: This is Hacking, need a proper fix.
            divisor = 3 if raw_info['arch'] in ['gfx950', 'gfx942'] else 2
            key = (raw_info['arch'], raw_info['_debug_task_id'] // divisor, 'op_attn_bwd')
        else:
            key = (raw_info['arch'], raw_info['_debug_task_id'], raw_info['kernel_name'])
        if key not in self.pkr_database:
            self.pkr_database[key] = pkr_factory(key)
        self.pkr_database[key].collect(raw_info)

    def aggregation_results(self):
        warned = False
        for pkr in self.pkr_database.values():
            pkr.conclude()
            opti = pkr.get_optimal_kernel(self._args.fudge_factor_tolerance, self._args.max_fudge_factor, allow_no_acceptable=True)
            if opti is None:
                if not warned:
                    print('\nAcceptables is empty. Re-run tests on missing entries\n')
                    warned = True
                print("\nTUNE_FLASH --entry_from_json Item: ", pkr.entry_from_json())
                continue
            pkr.update_max_fudge_factor(opti)
            yield opti

    def sancheck(self, fout):
        need_rerun = set()
        for pkr in self.pkr_database.values():
            pkr.conclude()
            opti = pkr.get_optimal_kernel(self._args.fudge_factor_tolerance, self._args.max_fudge_factor, allow_no_acceptable=True)
            if opti is None:
                need_rerun.add(pkr.tid)
        sc_report = {
            'need_rerun' : list(need_rerun),
        }
        json.dump(sc_report, fout, indent=2)

def do_main(args, db, fin):
    if fin.seekable():
        fin.seek(0, os.SEEK_END)
        fin_size = fin.tell()
        fin.seek(0)
        print(f'{fin_size=}')
    else:
        fin_size = 0
    if args.action == 'pipejson' or args.action == 'createtableonly':
        # FIXME: support pipes and streaming file with json_stream
        if args.action == 'createtableonly':
            create_table_only = True
        else:
            create_table_only = False
        for line in fin:
            db.upsert(line, create_table_only=create_table_only)
        print("[table_tool] Input closed, exiting", file=sys.stderr)
    elif args.action in ['rawjson', 'rawjson_fudge_check', 'opjson']:
        db.init_aggregation()
        pbar = tqdm(total=fin_size, desc='Processed bytes')
        for line in fin:
            db.aggregate(line)
            pbar.update(len(line))  # FIXME: support UTF-8 which len(line) != number of bytes
        if args.action == 'rawjson_fudge_check':
            for rawjson in db.aggregation_results():
                pass
        else:
            pbar = tqdm(total=len(db.pkr_database), desc='Processed kernels')
            with db._conn:  # Transaction
                for rawjson in db.aggregation_results():
                    if rawjson is None:
                        continue
                    db.upsert_json(rawjson, create_table_only=False)
                    # Dispatcher v3 should nullified such cases
                    # if 'CAUSAL' in rawjson['inputs']:
                    #     causal = rawjson['inputs']['CAUSAL']
                    # elif 'CAUSAL_TYPE' in rawjson['inputs']:
                    #     causal = rawjson['inputs']['CAUSAL_TYPE']
                    # else:
                    #     causal = False
                    ## Handles CAUSAL=True and BIAS_TYPE=1 case
                    ## No real use cases, just let the build system compile things
                    #if causal == True and rawjson['inputs']['BIAS_TYPE'] == 0:
                    #    rj2 = deepcopy(rawjson)
                    #    rj2['inputs']['BIAS_TYPE'] = 1
                    #    db.upsert_json(rj2, create_table_only=False)
                    pbar.update(1)
        for klass in KERNEL_NAME_TO_FACTORY.values():
            print(f'{klass.KERNEL_NAME=} {klass.KERNEL_MAX_FUDGE_FACTORS=}')
    elif args.action == 'rawsc':
        db.init_aggregation()
        pbar = tqdm(total=fin_size, desc='Processed bytes')
        for line in fin:
            db.aggregate(line)
            pbar.update(len(line))  # FIXME: support UTF-8 which len(line) != number of bytes
        with open(args.sc_report, 'w') as fout:
            db.sancheck(fout)
    elif args.action == 'dumpcsv':
        db.dumpcsv(args.table_name, args.table_file)
    elif args.action == 'loadcsv':
        db.loadcsv(args.table_file, args.table_name)
    db.close()

class OpDatabase(TuningDatabase):
    OPTABLE = True

    def get_table_name(self, tune_info: dict) -> str:
        kn = tune_info['kernel_name']
        return f'{self._args.kernel_family}${kn}'

    # Don't
    def ensure_table(self, tune_info : dict) -> str:
        return self.get_table_name(tune_info)

    def _create_table(self, tune_info):
        pass

def main():
    args = parse()
    if args.action in ['opjson']:
        db = OpDatabase(args)
    else:
        db = TuningDatabase(args)
    if args.i is not None:
        with open(args.i) as f:
            do_main(args, db, f)
    else:
        do_main(args, db, sys.stdin)

if __name__ == '__main__':
    main()
