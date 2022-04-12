package io.kinference.ort_gpu.data.utils

import ai.onnxruntime.*
import io.kinference.ort_gpu.ORTGPUData
import io.kinference.ort_gpu.data.map.ORTGPUMap
import io.kinference.ort_gpu.data.seq.ORTGPUSequence
import io.kinference.ort_gpu.data.tensor.ORTGPUTensor

inline fun <reified V : OnnxValue> createORTData(name: String?, data: V) : ORTGPUData<*> = when (data) {
    is OnnxTensor -> ORTGPUTensor(name, data)
    is OnnxMap -> ORTGPUMap(name, data)
    is OnnxSequence -> ORTGPUSequence(name, data)
    else -> error("Cannot find corresponding ONNXData type for ${data::class}")
}
