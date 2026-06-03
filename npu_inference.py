"""
RK3588 NPU Inference — ctypes wrapper around librknnrt.so
Bypasses rknn-toolkit-lite2's broken platform detection ("Unsupported run platform: Linux aarch64").

Benchmark: ~19.6 ms/inference on RK3588 with 3 NPU cores (vs ~100-300ms ONNX CPU).

Usage:
    from npu_inference import NPUInference
    npu = NPUInference('models/sprite_detector.rknn')
    output = npu.run(input_array)  # input: uint8 [1, 640, 640, 3] NHWC
    npu.release()
"""

import ctypes as ct
import numpy as np
import os
import time

LIBRKNNRT = "/usr/lib/librknnrt.so"

# Query types
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2

# Core mask values (NPU_CORE_*)
NPU_CORE_AUTO = 0
NPU_CORE_0 = 1
NPU_CORE_1 = 2
NPU_CORE_2 = 4
NPU_CORE_0_1 = 3
NPU_CORE_0_1_2 = 7


class _rknn_input(ct.Structure):
    _fields_ = [
        ("index", ct.c_uint),
        ("buf", ct.c_void_p),
        ("size", ct.c_uint),
        ("pass_through", ct.c_ubyte),
        ("type", ct.c_int),
        ("fmt", ct.c_int),
    ]


class _rknn_output(ct.Structure):
    _fields_ = [
        ("want_float", ct.c_ubyte),
        ("is_prealloc", ct.c_ubyte),
        ("index", ct.c_uint),
        ("buf", ct.c_void_p),
        ("size", ct.c_uint),
    ]


class NPUInference:
    """Minimal, reliable RK3588 NPU inference via ctypes.

    Handles:
      - Model loading (rknn_init)
      - I/O tensor setup (rknn_inputs_set / rknn_outputs_get)
      - Multi-core NPU (rknn_set_core_mask)
      - Cleanup (rknn_destroy)
    """

    def __init__(self, model_path: str, core_mask: int = NPU_CORE_0_1_2):
        self._lib = ct.CDLL(LIBRKNNRT)
        self._setup_signatures()

        self._ctx = ct.c_void_p(0)

        with open(model_path, "rb") as f:
            model_bytes = f.read()
        self._model_buf = (ct.c_ubyte * len(model_bytes)).from_buffer_copy(model_bytes)

        ret = self._lib.rknn_init(
            ct.byref(self._ctx), self._model_buf, len(model_bytes), 0, None
        )
        if ret != 0:
            raise RuntimeError(f"rknn_init failed: {ret}")

        # Enable multi-core NPU
        if core_mask != NPU_CORE_AUTO:
            self._lib.rknn_set_core_mask(self._ctx, core_mask)

        # Query I/O
        io_buf = (ct.c_ubyte * 8)()
        self._lib.rknn_query(self._ctx, RKNN_QUERY_IN_OUT_NUM, io_buf, 8)
        self.n_inputs = ct.c_uint.from_buffer(io_buf, 0).value
        self.n_outputs = ct.c_uint.from_buffer(io_buf, 4).value

        # Query attributes
        self._input_attr = self._query_attr(RKNN_QUERY_INPUT_ATTR, 0)
        self._output_attrs = [self._query_attr(RKNN_QUERY_OUTPUT_ATTR, i)
                              for i in range(self.n_outputs)]

        # Pre-allocate output buffer and rknn_output struct
        out_attr = self._output_attrs[0]
        self._output_buf = np.zeros(out_attr['dims'], dtype=np.float32)
        self._rknn_out = _rknn_output()
        self._rknn_out.want_float = 1
        self._rknn_out.is_prealloc = 1
        self._rknn_out.index = 0
        self._rknn_out.buf = self._output_buf.ctypes.data_as(ct.c_void_p)
        self._rknn_out.size = self._output_buf.nbytes

    def _setup_signatures(self):
        L = self._lib
        L.rknn_init.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p]
        L.rknn_init.restype = ct.c_int
        L.rknn_destroy.argtypes = [ct.c_void_p]
        L.rknn_destroy.restype = ct.c_int
        L.rknn_query.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_uint]
        L.rknn_query.restype = ct.c_int
        L.rknn_inputs_set.argtypes = [ct.c_void_p, ct.c_uint, ct.c_void_p]
        L.rknn_inputs_set.restype = ct.c_int
        L.rknn_run.argtypes = [ct.c_void_p, ct.c_void_p]
        L.rknn_run.restype = ct.c_int
        L.rknn_outputs_get.argtypes = [ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_void_p]
        L.rknn_outputs_get.restype = ct.c_int
        L.rknn_set_core_mask.argtypes = [ct.c_void_p, ct.c_int]
        L.rknn_set_core_mask.restype = ct.c_int

    def _query_attr(self, query_type: int, index: int) -> dict:
        """Query tensor attributes. Struct is 376 bytes (see module docstring)."""
        buf = (ct.c_ubyte * 376)()
        ct.c_uint.from_buffer(buf, 0).value = index
        ret = self._lib.rknn_query(self._ctx, query_type, buf, 376)
        if ret != 0:
            raise RuntimeError(f"rknn_query(type={query_type}, idx={index}) failed: {ret}")

        n_dims = ct.c_uint.from_buffer(buf, 4).value
        dims = tuple(ct.c_uint.from_buffer(buf, 8 + i * 4).value for i in range(n_dims))

        # Name at offset 40, 256 bytes, strip all nulls
        raw_name = bytes(buf[40:296])
        name = raw_name.replace(b'\x00', b'').decode('utf-8', errors='replace')

        return {
            'index': ct.c_uint.from_buffer(buf, 0).value,
            'n_dims': n_dims,
            'dims': dims,
            'name': name,
            'fmt': ct.c_int.from_buffer(buf, 304).value,
            'dtype': ct.c_int.from_buffer(buf, 308).value,
            'zp': ct.c_int.from_buffer(buf, 320).value,
            'scale': ct.c_float.from_buffer(buf, 324).value,
        }

    @property
    def input_shape(self) -> tuple:
        return self._input_attr['dims']

    @property
    def output_shape(self) -> tuple:
        return self._output_attrs[0]['dims']

    @property
    def input_name(self) -> str:
        return self._input_attr['name']

    def run(self, input_array: np.ndarray) -> np.ndarray:
        """Run NPU inference.

        Args:
            input_array: numpy array matching input_shape.
                         For int8 models: uint8 NHWC
                         For float models: float32 in model's native layout

        Returns:
            float32 numpy array with output_shape.
        """
        if not input_array.flags['C_CONTIGUOUS']:
            input_array = np.ascontiguousarray(input_array)

        # Determine input type from the array dtype
        if input_array.dtype == np.uint8:
            inp_type = 2   # UINT8 (RKNN_TENSOR_UINT8 = 2)
        elif input_array.dtype == np.float32:
            inp_type = 0   # FLOAT32 (RKNN_TENSOR_FLOAT32 = 0)
        elif input_array.dtype == np.float16:
            inp_type = 1   # FLOAT16
        else:
            inp_type = 0   # Default to float32

        # Determine format: if channels-last (NHWC), fmt=1; if NCHW, fmt=0
        # For NHWC: shape is (N, H, W, C); for NCHW: shape is (N, C, H, W)
        shape = input_array.shape
        if len(shape) == 4 and shape[-1] in (3, 4):
            inp_fmt = 1  # NHWC
        else:
            inp_fmt = 0  # NCHW

        inp = _rknn_input()
        inp.index = 0
        inp.buf = input_array.ctypes.data_as(ct.c_void_p)
        inp.size = input_array.nbytes
        inp.pass_through = 0
        inp.type = inp_type
        inp.fmt = inp_fmt

        self._lib.rknn_inputs_set(self._ctx, 1, ct.byref(inp))
        self._lib.rknn_run(self._ctx, None)
        self._lib.rknn_outputs_get(self._ctx, 1, ct.byref(self._rknn_out), None)

        return self._output_buf

    def release(self):
        if self._ctx:
            self._lib.rknn_destroy(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "models", "sprite_detector.rknn")

    with NPUInference(model_path) as npu:
        print(f"Model: {model_path}")
        print(f"Input:  {npu.input_name} → {npu.input_shape}")
        print(f"Output: {npu.output_shape}")

        # Warm-up
        dummy = np.random.randint(0, 255, npu.input_shape, dtype=np.uint8)
        _ = npu.run(dummy)

        # Benchmark
        times = []
        for i in range(100):
            t0 = time.perf_counter()
            _ = npu.run(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg = np.mean(times[10:])
        print(f"\n{'='*50}")
        print(f"  NPU Inference (RK3588, 3 cores)")
        print(f"  Average:  {avg:.2f} ms")
        print(f"  Min/Max:  {np.min(times[10:]):.2f} / {np.max(times[10:]):.2f} ms")
        print(f"  FPS:      {1000/avg:.0f}")
        print(f"{'='*50}")
