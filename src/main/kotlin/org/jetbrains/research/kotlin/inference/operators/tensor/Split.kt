package org.jetbrains.research.kotlin.inference.operators.tensor

import AttributeProto
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.splitWithAxis
import org.jetbrains.research.kotlin.inference.extensions.primitives.toIntArray
import org.jetbrains.research.kotlin.inference.operators.*

class Split(attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int) : Operator<Tensor, Tensor>(INFO, usedOutputsNum, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 0L),
            AttributeInfo("split", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true))

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "outputs"))

        private val INFO = OperatorInfo("Split", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(inputs: List<Tensor>): List<Tensor> {
        val axis = getAttributeValue("axis") as Long

        return when (val parts = getAttributeValueOrNull("split")) {
            null -> inputs.first().data.splitWithAxis(usedOutputsNum, axis.toInt()).map { it.asTensor("") }
            is Number -> inputs.first().data.splitWithAxis(parts.toInt(), axis.toInt()).map { it.asTensor("") }
            is List<*> -> inputs.first().data.splitWithAxis((parts as List<Long>).toIntArray(), axis.toInt()).toList().map { it.asTensor("") }
            else -> error("Unsupported value type")
        }
    }
}
