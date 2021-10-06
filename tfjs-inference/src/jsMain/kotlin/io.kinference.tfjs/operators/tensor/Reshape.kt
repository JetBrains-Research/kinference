package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.custom_externals.extensions.*
import io.kinference.tfjs.data.tensors.Tensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*
import io.kinference.protobuf.message.TensorProto

class Reshape(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "reshaped", optional = false, differentiable = true))

        private val INFO = OperatorInfo("Reshape", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val output = tidy {
            val input = inputs[0]!!.data
            val shape = inputs[1]!!.data

            val shapeData = shape.dataInt()

            for ((i, axisShape) in shapeData.withIndex()) {
                if (axisShape == 0) shapeData[i] = input.shape[i]
            }

            val negativeIdx = shapeData.indexOf(-1)
            if (negativeIdx != -1) {
                val elementsCount = shapeData.filter { it != -1 }.fold(1, Int::times)
                shapeData[negativeIdx] = input.size / elementsCount
            }
            return@tidy arrayOf(input.reshape(shapeData.toTypedArray()))
        }

        return listOf(output[0].asTensor("reshaped"))
    }
}
