package org.jetbrains.research.kotlin.mpp.inference.operators.tensor

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.concatenate

class Concat(attributes: Map<String, Attribute<Any>>) : Operator<Tensor, Tensor>("Concat", attributes, emptyList(), INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(InputInfo(0, TYPE_CONSTRAINTS, "inputs", true))

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "concat_result"))
    }

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<Tensor> {
        val axis = attributes["axis"]?.value as? Long ?: 0L

        return listOf(inputs.concatenate(axis.toInt()))
    }
}
