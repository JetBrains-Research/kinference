package org.jetbrains.research.kotlin.mpp.inference.operators.tensor

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor

class Transpose(attributes: Map<String, Attribute<Any>>) : Operator<Tensor, Tensor>("Transpose", attributes, emptyList(), INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(InputInfo(0, TYPE_CONSTRAINTS, "data", true))

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "transposed"))
    }

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<Tensor> {
        return listOf(inputs.first().transpose(attributes["perm"]?.value as? List<Long>))
    }
}
