package org.jetbrains.research.kotlin.inference.operators.tensor

import AttributeProto
import TensorProto
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.operators.*

class Gather(attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int = 1) : Operator<Tensor, Tensor>(INFO, usedOutputsNum, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            InputInfo(0, TYPE_CONSTRAINTS, "data", true),
            InputInfo(1, setOf(TensorProto.DataType.INT32, TensorProto.DataType.INT64), "indices", true)
        )

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))

        private val INFO = OperatorInfo("Gather", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(inputs: List<Tensor>): List<Tensor> {
        val (data, indices) = inputs
        val axis = (data).data.indexAxis((getAttributeValue("axis") as Number).toInt())
        return listOf(data.data.gather(indices.data, axis).asTensor(""))
    }
}
