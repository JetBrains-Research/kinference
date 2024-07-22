package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.toIntArray

sealed class Unsqueeze(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Unsqueeze {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in UnsqueezeVer1.VERSION.asRange() -> UnsqueezeVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Unsqueeze operator: $version")
            }
        }
    }
}

class UnsqueezeVer1 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Unsqueeze(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axes", setOf(AttributeProto.AttributeType.INTS), required = false, default = null)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "axes", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "expanded", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Unsqueeze", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axes: IntArray? by attributeOrNull { it: LongArray? -> it?.toIntArray() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data
        val axesInput = inputs.getOrNull(1)?.data as? LongNDArray
        val axes = axes ?: axesInput?.array?.toArray()?.toIntArray()

        requireNotNull(axes) { "Unsqueeze axes should be specified in either operator attribute or input list" }
        require(axes.size == axes.toSet().size) { "Axes must contains only unique elements, present: ${axes.joinToString(prefix = "[", postfix = "]")}" }

        return listOf(input.unsqueeze(*axes).asTensor("expanded"))
    }
}
