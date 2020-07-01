package org.jetbrains.research.kotlin.mpp.inference.operators.tensor

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.toIntArray

class Squeeze(attributes: Map<String, Attribute<Any>>) : Operator<Tensor, Tensor>("Squeeze", attributes, ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axes", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(InputInfo(0, TYPE_CONSTRAINTS, "data", true))

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "squeezed"))
    }

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<Tensor> {
        val axes = (getAttributeValueOrNull("axes") as? List<Long>) ?: emptyList()
        return listOf(inputs.first().squeeze(*axes.toIntArray()))
    }
}
