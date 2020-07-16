package org.jetbrains.research.kotlin.mpp.inference.operators.seq

import AttributeProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.seq.TensorSeq
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.concatenate
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.stack
import org.jetbrains.research.kotlin.mpp.inference.operators.*

class ConcatFromSequence(attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int)
    : Operator<TensorSeq, Tensor>(INFO, usedOutputsNum, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("new_axis", setOf(AttributeProto.AttributeType.INT), false, default = 0L)
        )

        private val INPUTS_INFO = listOf(InputInfo(0, TYPE_CONSTRAINTS, "input_sequence", true))

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "concat_result"))

        private val INFO = OperatorInfo("ConcatFromSequence", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(inputs: List<TensorSeq>): List<Tensor> {
        val axis = getAttributeValue("axis") as Long
        val newAxis = getAttributeValue("new_axis") as Long

        val srcTensors = inputs.first().data
        val tensor = if (newAxis == 1L) srcTensors.stack(axis.toInt()) else srcTensors.concatenate(axis.toInt())
        return listOf(tensor)
    }
}
