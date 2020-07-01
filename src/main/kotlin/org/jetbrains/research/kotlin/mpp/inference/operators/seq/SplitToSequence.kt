package org.jetbrains.research.kotlin.mpp.inference.operators.seq

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.seq.TensorSeq
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.*
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.types.SequenceInfo

class SplitToSequence(attributes: Map<String, Attribute<Any>>) : Operator<Tensor, TensorSeq>("SplitToSequence", attributes, emptyList(), INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            InputInfo(0, TYPE_CONSTRAINTS, "input", true),
            InputInfo(1, setOf(TensorProto.DataType.INT64, TensorProto.DataType.INT32), "split", false)
        )

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output_sequence"))
    }

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<TensorSeq> {
        val axis = attributes["axis"]?.value as? Long ?: 0L

        val tensors = when (val parts = attributes["split"]?.value) {
            null -> inputs.first().splitWithAxis(numOutputs, axis.toInt())
            is Number -> inputs.first().splitWithAxis(parts.toInt(), axis.toInt())
            is List<*> -> inputs.first().splitWithAxis((parts as List<Long>).toIntArray(), axis.toInt())
            else -> error("Unsupported splitter value type")
        }

        return listOf(TensorSeq(tensors, SequenceInfo("output_sequence", tensors.first().info.type)))
    }
}
