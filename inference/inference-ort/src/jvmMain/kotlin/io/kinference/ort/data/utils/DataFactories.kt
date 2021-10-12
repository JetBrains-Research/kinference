package io.kinference.ort.data.utils

import ai.onnxruntime.*
import io.kinference.data.ONNXData
import io.kinference.ort.data.map.ORTMap
import io.kinference.ort.data.seq.ORTSequence
import io.kinference.ort.data.tensor.ORTTensor

inline fun <reified V : OnnxValue> createORTData(name: String?, data: V) : ONNXData<*> = when (data) {
    is OnnxTensor -> ORTTensor(name, data)
    is OnnxMap -> ORTMap(name, data)
    is OnnxSequence -> ORTSequence(name, data)
    else -> error("")
}
