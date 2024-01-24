#include <aotriton/dtypes.h>
#include <aotriton/util.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace pyaotriton {
    namespace v2 {
        namespace flash {
            void setup_module(py::module_& m) {
                m.def("attn_fwd", &aotriton::v2::flash::attn_fwd, "Flash Attention Forward Pass",
                      py::arg("q"),
                      py::arg("k"),
                      py::arg("v"),
                      py::arg("sm_scale"),
                      py::arg("softmax_lse"),
                      py::arg("out"),
                      py::arg("dropout_p"),
                      py::arg("philox_seed"),
                      py::arg("philox_offset"),
                      py::arg("encoded_softmax"),
                      py::arg("is_causal"),
                      py::arg("stream") = nullptr
                );
                m.def("attn_bwd", &aotriton::v2::flash::attn_bwd, "Flash Attention Backward Pass",
                      py::arg("q"),
                      py::arg("k"),
                      py::arg("v"),
                      py::arg("sm_scale"),
                      py::arg("out"),
                      py::arg("dout"),
                      py::arg("dq"),
                      py::arg("dk"),
                      py::arg("dv"),
                      py::arg("softmax_lse"),
                      py::arg("delta"),
                      py::arg("dropout_p"),
                      py::arg("philox_seed"),
                      py::arg("philox_offset"),
                      py::arg("is_causal"),
                      py::arg("stream") = nullptr
                );
            }
        }

        void setup_module(py::module_& m) {
            py::module_ mod_flash = m.def_submodule("flash", "Flash Attention API");
            flash::setup_module(mod_flash);
        }
    }

    void def_stream(py::module_& m) {
        py::class_<aotriton::Stream>(m, "Stream")
            .def(py::init<>())
        ;
    }

    void def_dtype(py::module_& m) {
#define EV(name)    value(#name, aotriton::DType::name)
        py::enum_<aotriton::DType>(m, "DType")
            .EV(kUnknown)
            .EV(kFloat32)
            .EV(kFloat16)
            .EV(kBFloat16)
            .EV(kInt8)
            .EV(kInt16)
            .EV(kInt32)
            .EV(kInt64)
            .EV(kUInt8)
            .EV(kUInt16)
            .EV(kUInt32)
            .EV(kUInt64)
            .export_values()
        ;
#undef EV
    }

    void def_hipruntime(py::module_& m);

    template<int Rank>
    void def_tensorview(py::module_& m, const std::string& name) {
        py::class_<aotriton::TensorView<Rank>>(m, name.c_str())
            .def(py::init<intptr_t,
                          std::array<uint64_t, Rank>,
                          std::array<uint64_t, Rank>,
                          aotriton::DType>())
            .def("size", &aotriton::TensorView<Rank>::size)
            .def("stride", &aotriton::TensorView<Rank>::stride)
            .def_property_readonly("sizes", &aotriton::TensorView<Rank>::sizes)
            .def_property_readonly("strides", &aotriton::TensorView<Rank>::strides)
            .def_property_readonly("data_ptr", &aotriton::TensorView<Rank>::data_ptr)
            .def_property_readonly("dtype", &aotriton::TensorView<Rank>::dtype)
        ;
    }

    void setup_module(py::module_& m) {
        m.doc() = "AOTriton Python binding";
        def_stream(m);
        def_dtype(m);
        def_hipruntime(m);
        def_tensorview<4>(m, "T4");
        def_tensorview<2>(m, "T2");
        def_tensorview<1>(m, "T1");
        py::module_ mod_v2api = m.def_submodule("v2", "v2 API namespace");
        v2::setup_module(mod_v2api);
    }

}

PYBIND11_MODULE(pyaotriton, m) {
    pyaotriton::setup_module(m);
}
