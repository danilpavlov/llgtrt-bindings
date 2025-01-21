#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../tlc.h"  // Include your C++ header for tlc

#include "tensorrt_llm/executor/executor.h" 

namespace py = pybind11;

PYBIND11_MODULE(trtllm_c_bindings, m) {
    m.doc() = "Python bindings for trtllm-c library";

    // Expose TlcInitParams struct
    py::class_<TlcInitParams>(m, "TlcInitParams")
        .def(py::init<>())
        .def_readwrite("engine_path", &TlcInitParams::engine_path)
        // Expose other members as needed
        ;

    // Expose TlcRequestParams struct
    py::class_<TlcRequestParams>(m, "TlcRequestParams")
        .def(py::init<>())
        .def_readwrite("max_new_tokens", &TlcRequestParams::max_new_tokens)
        // Expose other members as needed
        ;

    // Expose TlcRequest struct
    py::class_<TlcRequest>(m, "TlcRequest")
        .def(py::init<>())
        .def_readwrite("tokens", &TlcRequest::tokens)
        .def_readwrite("num_tokens", &TlcRequest::num_tokens)
        .def_readwrite("client_req_id", &TlcRequest::client_req_id)
        .def_readwrite("params", &TlcRequest::params)
        ;

    // Expose TlcResponse struct
    py::class_<TlcResponse>(m, "TlcResponse")
        .def_readonly("req_id", &TlcResponse::req_id)
        .def_readonly("sequence_idx", &TlcResponse::sequence_idx)
        .def_readonly("is_seq_final", &TlcResponse::is_seq_final)
        .def_readonly("is_req_final", &TlcResponse::is_req_final)
        .def_readonly("finish_reason", &TlcResponse::finish_reason)
        .def_readonly("error", &TlcResponse::error)
        .def_readonly("tokens", &TlcResponse::tokens)
        .def_readonly("num_tokens", &TlcResponse::num_tokens)
        // Expose other members as needed
        ;

    // Expose the TlcExecutor class (if you have a C++ class)
    py::class_<TlcExecutor>(m, "TlcExecutor")
        .def("can_enqueue_request", [](TlcExecutor* self) {
            return tlc_can_enqueue_request(self);
        });

    // Expose functions
    m.def("tlc_default_init_params", &tlc_default_init_params, "Set default initialization parameters");
    m.def("tlc_init", [](const TlcInitParams& params) -> TlcExecutor* {
        TlcExecutor* executor = nullptr;
        //TlcExecutor* executor = new TlcExecutor();
        
        TlcStatus status = tlc_init(&params, &executor);
        if (status) {
            throw std::runtime_error(status);
        }
        return executor;
    }, "Initialize the TLC Executor");

    m.def("tlc_shutdown", [](TlcExecutor* executor) {
        tlc_shutdown(executor);
    }, "Shutdown the TLC Executor");

    m.def("tlc_enqueue_request", [](TlcExecutor* executor, const TlcRequest& request) -> TlcReqId {
        TlcReqId req_id;
        TlcStatus status = tlc_enqueue_request(executor, &request, &req_id);
        if (status) {
            throw std::runtime_error(status);
        }
        return req_id;
    }, "Enqueue a request");

    m.def("tlc_await_responses", [](TlcExecutor* executor, uint32_t timeout_ms) -> std::vector<TlcResponse> {
        const TlcResponse* responses_ptr = nullptr;
        uint32_t num_responses = 0;
        TlcStatus status = tlc_await_responses(executor, timeout_ms, &responses_ptr, &num_responses);
        if (status) {
            throw std::runtime_error(status);
        }
        std::vector<TlcResponse> responses(responses_ptr, responses_ptr + num_responses);
        return responses;
    }, "Await responses");

    // Expose other functions as needed
}