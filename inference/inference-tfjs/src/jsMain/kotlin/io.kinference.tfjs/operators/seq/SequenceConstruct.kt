package io.kinference.tfjs.operators.seq

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.types.ValueTypeInfo

sealed class SequenceConstruct(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSSequence>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): SequenceConstruct {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SequenceConstructVer11.VERSION.asRange() -> SequenceConstructVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of SequenceConstruct operator: $version")
            }
        }
    }
}


class SequenceConstructVer11 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : SequenceConstruct(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(VariadicIOInfo(0, ALL_DATA_TYPES, "inputs", minimumArity = 1, heterogeneous = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output_sequence", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("SequenceConstruct", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSSequence?> {
        val srcTensors = inputs.filterNotNull()
        val seq = TFJSSequence(
            name = "output_sequence",
            data = srcTensors,
            info = ValueTypeInfo.SequenceTypeInfo(
                elementType = ValueTypeInfo.TensorTypeInfo(
                    type = srcTensors.first().info.type
                )
            )
        )

        return listOf(seq)
    }
}
