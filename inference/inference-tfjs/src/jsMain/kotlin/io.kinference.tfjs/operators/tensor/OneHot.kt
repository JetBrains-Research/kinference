package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.dataInt
import io.kinference.ndarray.extensions.oneHot
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class OneHot(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): OneHot {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in OneHotVer9.VERSION.asRange() -> OneHotVer9(name, attributes, inputs, outputs)
                else -> error("Unsupported version of OneHot operator: $version")
            }
        }
    }
}

class OneHotVer9 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : OneHot(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, NUMBER_DATA_TYPES, "indices", optional = false, differentiable = false),
            IOInfo(1, NUMBER_DATA_TYPES, "depth", optional = false, differentiable = false),
            IOInfo(2, ALL_DATA_TYPES, "values", optional = true)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), required = false, default = -1L)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false, differentiable = false))

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("OneHot", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val depth = inputs[1]!!.data.dataInt()[0]
        val indices = inputs[0]!!.data as NumberNDArrayTFJS
        val values = inputs[2]!!.data
        return listOf(NDArrayTFJS.oneHot(indices, depth, values, axis).asTensor("output"))
    }
}
