package io.kinference.tfjs.operators.seq

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.types.ValueTypeInfo

sealed class SequenceErase(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSData<*>, TFJSSequence>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): SequenceErase {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SequenceEraseVer11.VERSION.asRange() -> SequenceEraseVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of SequenceErase operator: $version")
            }
        }
    }
}


class SequenceEraseVer11 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : SequenceErase(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input_sequence", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE),
            IOInfo(0, setOf(TensorProto.DataType.INT64, TensorProto.DataType.INT32), "position", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "output_sequence", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("SequenceErase", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSData<*>?>): List<TFJSSequence> {
        val seq = inputs[0]!! as TFJSSequence
        val seqList = ArrayList(seq.data)
        val positionTensor = inputs.getOrNull(1)?.data as? NumberNDArrayTFJS

        val position = positionTensor?.singleValue()?.toInt() ?: (seqList.size - 1)
        val actualPosition = if (position >= 0) position else seqList.size + position

        require(actualPosition >= 0 && actualPosition < seqList.size) { "Index $position is out of range [-${seqList.size}, ${seqList.size} - 1]" }

        seqList.removeAt(actualPosition)
        val outputSeq = TFJSSequence(
            name = "output_sequence",
            data = seqList,
            info = ValueTypeInfo.SequenceTypeInfo(
                elementType = seq.info.elementType
            )
        )
        return listOf(outputSeq)
    }
}
