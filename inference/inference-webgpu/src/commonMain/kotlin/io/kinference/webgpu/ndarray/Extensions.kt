package io.kinference.webgpu.ndarray

import io.kinference.types.TensorShape
import io.kinference.types.ValueTypeInfo
import io.kinference.webgpu.tensor.WebGPUTensor

fun NDArray.asTensor(name: String? = null) =
    WebGPUTensor(name,this, ValueTypeInfo.TensorTypeInfo(TensorShape(info.shape), info.type.resolve()))
