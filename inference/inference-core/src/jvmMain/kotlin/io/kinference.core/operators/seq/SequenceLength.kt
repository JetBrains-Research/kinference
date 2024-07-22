package io.kinference.core.operators.seq

import io.kinference.attribute.Attribute
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto

sealed class SequenceLength(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KIONNXSequence, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): SequenceLength {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SequenceLengthVer11.VERSION.asRange() -> SequenceLengthVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of SequenceLength operator: $version")
            }
        }
    }
}


class SequenceLengthVer11 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : SequenceLength(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input_sequence", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE))

        private val OUTPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "length", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("SequenceLength", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KIONNXSequence?>): List<KITensor?> {
        val seq = inputs.first()!!.data
        val seqLength = LongNDArray.scalar(seq.size.toLong())
        return listOf(seqLength.asTensor("length"))
    }
}
