package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.custom_externals.core.tensor
import io.kinference.custom_externals.extensions.tidy
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.operators.*
import io.kinference.protobuf.message.TensorProto

class Shape(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false))

        private val INFO = OperatorInfo("Shape", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }


    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val outputs = tidy {
            val input = inputs[0]!!
            val inputShape = input.data.shape
            return@tidy arrayOf(tensor(inputShape, arrayOf(inputShape.size), "int32"))
        }

        return listOf(outputs[0].asTensor("shape"))
    }
}
