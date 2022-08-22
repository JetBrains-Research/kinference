package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.extensions.*

sealed class Expand(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 8)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ExpandVer8.VERSION.asRange() -> ExpandVer8(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Expand operator: $version")
            }
    }
}

class ExpandVer8(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Expand(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "input", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false, differentiable = true))

        internal val VERSION = VersionInfo(sinceVersion = 8)
        private val INFO = OperatorInfo("Expand", emptySet(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val output = tidy {
            val input = inputs[0]!!.data
            val shape = inputs[1]!!.data

            val shapeArray = shape.dataInt()

            val broadcastedShape = Broadcasting.broadcastShape(listOf(input.shape.toIntArray(), shapeArray))

            return@tidy arrayOf(input.broadcastTo(broadcastedShape.toTypedArray()))
        }

        return listOf(output[0].asTensor("output"))
    }

}
