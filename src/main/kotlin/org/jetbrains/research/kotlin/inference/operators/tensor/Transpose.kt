package org.jetbrains.research.kotlin.inference.operators.tensor

import AttributeProto
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.operators.*

class Transpose(attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int) : Operator<Tensor, Tensor>(INFO, usedOutputsNum, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("perm", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(InputInfo(0, TYPE_CONSTRAINTS, "data", true))

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "transposed"))

        private val INFO = OperatorInfo("Transpose", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(inputs: List<Tensor>): List<Tensor> {
        val permutations = getAttributeValueOrNull("perm") as? List<Long>
        return listOf(inputs.first().data.transpose(permutations).asTensor())
    }
}
