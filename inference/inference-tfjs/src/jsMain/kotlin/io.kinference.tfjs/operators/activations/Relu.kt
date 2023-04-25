package io.kinference.tfjs.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.relu
import io.kinference.operator.*
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Relu(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6, untilVersion = 14)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Relu {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ReluVer6.VERSION.asRange() -> ReluVer6(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Relu operator: $version")
            }
        }
    }
}


class ReluVer6(
    name: String,
    attributes: Map<String, Attribute<Any>> = emptyMap(),
    inputs: List<String>,
    outputs: List<String>
) : Relu(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 6, untilVersion = 14)
        private val INFO = OperatorInfo("Relu", emptyMap(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val result = relu(inputs[0]!!.data as NumberNDArrayTFJS)
        return listOf(result.asTensor("Y"))
    }
}
