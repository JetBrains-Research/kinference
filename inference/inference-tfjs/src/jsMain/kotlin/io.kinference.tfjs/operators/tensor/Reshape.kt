package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.extensions.dataInt
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Reshape(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 5, untilVersion = 14)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ReshapeVer5.VERSION.asRange() -> ReshapeVer5(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Constant operator: $version")
            }
    }
}

class ReshapeVer5(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Reshape(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "reshaped", optional = false, differentiable = true))

        internal val VERSION = VersionInfo(sinceVersion = 5, untilVersion = 14)
        private val INFO = OperatorInfo("Reshape", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data
        val shape = inputs[1]!!.data
        val output = tidyNDArray {
            val shapeData = shape.dataInt().copyOf()

            for ((i, axisShape) in shapeData.withIndex()) {
                if (axisShape == 0) shapeData[i] = input.shape[i]
            }

            val negativeIdx = shapeData.indexOf(-1)
            if (negativeIdx != -1) {
                val elementsCount = shapeData.filter { it != -1 }.fold(1, Int::times)
                shapeData[negativeIdx] = input.linearSize / elementsCount
            }
            return@tidyNDArray input.reshape(shapeData)
        }

        return listOf(output.asTensor("reshaped"))
    }
}
