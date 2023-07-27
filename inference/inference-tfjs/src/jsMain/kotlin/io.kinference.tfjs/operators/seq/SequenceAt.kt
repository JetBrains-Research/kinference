package io.kinference.tfjs.operators.seq

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor

sealed class SequenceAt(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSData<*>, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): SequenceAt {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SequenceAtVer11.VERSION.asRange() -> SequenceAtVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of SequenceAt operator: $version")
            }
        }
    }
}


class SequenceAtVer11 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : SequenceAt(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input_sequence", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE),
            IOInfo(1, setOf(TensorProto.DataType.INT64, TensorProto.DataType.INT32), "position", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "tensor", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("SequenceAt", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSData<*>?>): List<TFJSTensor?> {
        val seq = ArrayList(inputs[0]!!.data as List<TFJSTensor>)
        val positionTensor = inputs[1]!!.data as NumberNDArrayTFJS

        val position = positionTensor.singleValue().toInt()
        val actualPosition = if (position >= 0) position else seq.size + position

        require(actualPosition >= 0 && actualPosition < seq.size) { "Index $position is out of range [-${seq.size}, ${seq.size} - 1]" }

        val elementAt = seq[actualPosition]
        val tensor = TFJSTensor(
            name = "tensor",
            data = elementAt.data.clone(),
            info = elementAt.info
        )
        return listOf(tensor)
    }
}
