package org.jetbrains.research.kotlin.mpp.inference.operators.tensor

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.*

class Split(attributes: Map<String, Attribute<Any>>) : Operator<Tensor, Tensor>("Split", attributes, emptyList(), INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true))

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "outputs"))
    }

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<Tensor> {
        val axis = attributes["axis"]?.value as? Long ?: 0L

        return when (val parts = attributes["split"]?.value) {
            null -> inputs.first().splitWithAxis(numOutputs, axis.toInt())
            is Number -> inputs.first().splitWithAxis(parts.toInt(), axis.toInt())
            is List<*> -> inputs.first().splitWithAxis((parts as List<Long>).toIntArray(), axis.toInt())
            else -> error("Unsupported splitter value type")
        }
    }
}
