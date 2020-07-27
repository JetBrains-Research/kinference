package org.jetbrains.research.kotlin.inference.operators.seq

import AttributeProto
import TensorProto
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.seq.TensorSeq
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.splitWithAxis
import org.jetbrains.research.kotlin.inference.operators.*
import org.jetbrains.research.kotlin.inference.types.SequenceInfo

class SplitToSequence(attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int)
    : Operator<Tensor, TensorSeq>(INFO, usedOutputsNum, attributes) {
    companion object {
        private const val DEFAULT_SPLIT_LENGTH = 1
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 0L),
            AttributeInfo("keepdims", setOf(AttributeProto.AttributeType.INT), false, default = 1L)
        )

        private val INPUTS_INFO = listOf(
            InputInfo(0, TYPE_CONSTRAINTS, "input", true),
            InputInfo(1, setOf(TensorProto.DataType.INT64, TensorProto.DataType.INT32), "split", false)
        )

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output_sequence"))

        private val INFO = OperatorInfo("SplitToSequence", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(inputs: List<Tensor>): List<TensorSeq> {
        val axis = getAttributeValue("axis") as Long
        val keepDims = getAttributeValue("keepdims") as Long
        val parts = inputs.elementAtOrNull(1)

        val tensors = if (parts == null) {
            inputs.first().data.splitWithAxis(inputs.first().data.shape[axis.toInt()], axis.toInt(), keepDims == 1L)
        } else {
            inputs.first().data.splitWithAxis(parts.data, axis.toInt())
        }

        return listOf(TensorSeq(tensors.map { it.asTensor("") }, SequenceInfo("output_sequence", tensors.first().type)))
    }
}
