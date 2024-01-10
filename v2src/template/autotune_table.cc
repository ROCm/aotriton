#include "[[shim_kernel_name]].h"
#include <limits>

namespace aotriton::v2::[[kernel_family_name]]::autotune {

// Store the real kernel, with a set of parameters
template<int GodelNumber>
struct Autotune_[[shim_kernel_name]] {
    void operator()([[param_class_name]]& params) {
    }
};

[[param_class_name]]_TunableKernel::TuningResult optimal_for = {
    [[autotune_optimals]]
};

[[param_class_name]]::AutoTuneTableEntry
[[param_class_name]]::autotune_table[][ [[number_of_functionals]] ] = {
    [[autotune_table_entries]]
};

[[autotune_table_entry_instances]];

}
