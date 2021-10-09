package io.kinference.ort.data

import ai.onnxruntime.*
import io.kinference.data.ONNXData
import io.kinference.ort.data.map.ORTMap
import io.kinference.ort.data.seq.ORTSequence
import io.kinference.ort.data.tensor.ORTTensor

abstract class ORTData(override val data: OnnxValue, override val name: String?) : ONNXData<OnnxValue> {
    companion object {
        inline operator fun <reified V : OnnxValue> invoke(name: String?, data: V) : ORTData = when (data) {
            is OnnxTensor -> ORTTensor(data, name)
            is OnnxMap -> ORTMap(data, name)
            is OnnxSequence -> ORTSequence(data, name)
            else -> error("")
        }
    }
}
