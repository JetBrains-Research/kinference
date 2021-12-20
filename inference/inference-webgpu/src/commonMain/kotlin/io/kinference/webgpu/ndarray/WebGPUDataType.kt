package io.kinference.webgpu.ndarray

import io.kinference.protobuf.message.TensorProto

enum class WebGPUDataType(val wgslType: String, val sizeBytes: Int) {
    INT32("i32", 4),
    UINT32("u32", 4),
    FLOAT32("f32", 4);
}

fun WebGPUDataType.resolve(): TensorProto.DataType = when (this) {
    WebGPUDataType.INT32 -> TensorProto.DataType.INT32
    WebGPUDataType.UINT32 -> TensorProto.DataType.UINT32
    WebGPUDataType.FLOAT32 -> TensorProto.DataType.FLOAT
}
