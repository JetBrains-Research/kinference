package io.kinference.webgpu.ndarray

import io.kinference.utils.webgpu.BufferData

fun BufferData.unpack(info: NDArrayInfo): TypedNDArrayData = when (info.type) {
    WebGPUDataType.INT32 -> IntNDArrayData(this.toIntArray())
    WebGPUDataType.UINT32 -> UIntNDArrayData(this.toUIntArray())
    WebGPUDataType.FLOAT32 -> FloatNDArrayData(this.toFloatArray())
}
