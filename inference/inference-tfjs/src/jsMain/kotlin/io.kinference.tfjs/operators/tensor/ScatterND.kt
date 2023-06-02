package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.extensions.tensorScatterUpdate
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class ScatterND(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11, untilVersion = 16)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): ScatterND {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ScatterNDVer11.VERSION.asRange() -> ScatterNDVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of ScatterND operator: $version")
            }
        }
    }
}

class ScatterNDVer11(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : ScatterND(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "indices", optional = false, differentiable = false),
            IOInfo(0, ALL_DATA_TYPES, "updates", optional = false, differentiable = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11, untilVersion = 16)
        private val INFO = OperatorInfo("ScatterND", emptyList(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data
        val indices = inputs[1]!!.data
        val updates = inputs[2]!!.data

        return listOf(input.tensorScatterUpdate(indices, updates).asTensor("output"))
    }
}
