package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.extensions.transpose
import io.kinference.onnx.AttributeProto
import io.kinference.operators.*

class Transpose(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("perm", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = true))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "transposed", optional = false, differentiable = true))

        private val INFO = OperatorInfo("Transpose", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val perm: List<Number>? by attributeOrNull()

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        return listOf(inputs.first()!!.data.toMutable().transpose(perm).asTensor())
    }
}
