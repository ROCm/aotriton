from collections import defaultdict

class ClusterKernel(object):
    def __init__(self):
        self._registry = []

    def collect_object_file(self, ofd : 'ObjectFileDescription'):
        self._registry.append(ofd)

    def calc_clustering_scheme(self, n_combination):
        cluster_by = {}
        if n_combination == 0:  # No need to change functional(s) to 'Any'
            dic = defaultdict(list)
            for ofd in self._registry:
                fonly = ofd.functional_signature + '_' + ofd.target_arch
                dic[fonly].append(ofd)
            cluster_by[None] = dic
            return cluster_by
        kdesc = self._registry[0]._triton_kernel_desc
        keys = []
        for m in kdesc._func_meta:
            if m.nchoices <= 1:
                continue
            keys.append(m.repr_name)
        for keycomb in itertools.combinations(keys, n_combination):
            cluster_by[keycomb] = defaultdict(list)
        for by, dic in cluster_by.items():
            if isinstance(by, str):
                sans = set([by])
            else:
                sans = set(by)
            for ofd in self._registry:
                ksig = ofd._signature
                fonly = ksig.get_partial_functional_signature(sans) + ksig._gpu
                dic[fonly].append(ofd)
        return cluster_by

class ClusterKernelFamily(object):
    def __init__(self):
        self._registry = defaultdict(ClusterKernel)

    def collect_object_file(self, ofd : 'ObjectFileDescription'):
        self._registry[ofd.SHIM_KERNEL_NAME].collect_object_file(ofd)

    def gen_clusters(self):
        for kernel_name, registry_0 in self._registry.items():
            yield kernel_name, registry_0

class ClusterRegistry(object):
    def __init__(self):
        self._registry = defaultdict(ClusterKernelFamily)

    def collect_object_file(self, ofd : 'ObjectFileDescription'):
        self._registry[ofd.KERNEL_FAMILY].collect_object_file(ofd)

    def gen_clusters(self, n_combination):
        for family, registry_0 in self._registry.items():
            for kernel_name, registry_1 in registry_0.gen_clusters():
                yield family, kernel_name, registry_1.calc_clustering_scheme(n_combination=n_combination)

    def write_clustering_tests(self, f):
        for family, kernel_name, cluster in self.gen_clusters(n_combination=2):
            print(f'mkdir -p {family}/{kernel_name}', file=f)
            for by, clusters in cluster.items():
                bypath = '-'.join(by)
                print(f'mkdir -p {family}/{kernel_name}/{bypath}', file=f)
                for fonly, obj_list in clusters.items():
                    tar = f'{family}/{kernel_name}/{bypath}/{fonly}.tar'
                    print(f'tar cf {tar} ', ' '.join([str(o.obj.absolute()) for o in obj_list]), file=f)
                    print(f'zstd {tar}', file=f)

    def write_bare(self, args, f):
        for family, kernel_name, cluster_bys in self.gen_clusters(n_combination=0):
            # cluster_bys[None]: only cluster psels and copts
            # Experiment shows it is not needed to cluster by one or more functionals.
            # XZ + clustering psels+copts is good enough
            clusters = cluster_bys[None]
            for fonly, obj_list in clusters.items():
                first_obj = obj_list[0]
                dir_arch = ARCH_TO_DIRECTORY[first_obj.target_arch]
                print(dir_arch, family, kernel_name, fonly, end=';', sep=';', file=f)
                path_list = [str(o.obj.absolute()) for o in obj_list]
                print(*path_list, sep=';', file=f)

