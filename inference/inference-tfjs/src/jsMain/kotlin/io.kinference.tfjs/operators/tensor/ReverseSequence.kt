package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class ReverseSequence(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ReverseSequenceVer10.VERSION.asRange() -> ReverseSequenceVer10(name, attributes, inputs, outputs)
                else -> error("Unsupported version of ReverseSequence operator: $version")
            }
    }
}


class ReverseSequenceVer10 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : ReverseSequence(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("batch_axis", setOf(AttributeProto.AttributeType.INT), default = 1L),
            AttributeInfo("time_axis", setOf(AttributeProto.AttributeType.INT), default = 0L),
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "input", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "sequence_lens", optional = false),
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "Y", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("ReverseSequence", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val batchAxis: Int by attribute("batch_axis") { it: Number? -> it?.toInt() ?: 1 }
    private val timeAxis: Int by attribute("time_axis") { it: Number? -> it?.toInt() ?: 0 }

    private val reverseMode = ReverseSeqMode.get(batchAxis, timeAxis)

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!
        require(input.info.type != TensorProto.DataType.STRING) { "String tensors are not supported by current version of ReverseSequence" }

        val seqLens = (inputs[1]!!.data as NumberNDArrayTFJS).dataInt()
        val reversed = input.data.reverseSeq(reverseMode, seqLens)

        return listOf(reversed.asTensor("Y"))
    }
}
