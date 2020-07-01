package org.jetbrains.research.kotlin.mpp.inference.operators.seq

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.seq.TensorSeq
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.*
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.types.SequenceInfo

class SplitToSequence(attributes: Map<String, Attribute<Any>>) : Operator<Tensor, TensorSeq>("SplitToSequence", attributes, emptyList(), INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private const val DEFAULT_SPLIT_LENGTH = 1
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            InputInfo(0, TYPE_CONSTRAINTS, "input", true),
            InputInfo(1, setOf(TensorProto.DataType.INT64, TensorProto.DataType.INT32), "split", false)
        )

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output_sequence"))
    }

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<TensorSeq> {
        val axis = attributes["axis"]?.value as? Long ?: 0L
        val keepDims = attributes["keepdims"]?.value as? Long ?: 1L
        val parts = inputs.elementAtOrNull(1)

        val tensors = if (parts == null) {
            inputs.first().splitWithAxis(DEFAULT_SPLIT_LENGTH, axis.toInt(), keepDims == 1L)
        } else {
            inputs.first().splitWithAxis(parts, axis.toInt())
        }

        return listOf(TensorSeq(tensors, SequenceInfo("output_sequence", tensors.first().info.type)))
    }
}
