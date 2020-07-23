package org.jetbrains.research.kotlin.mpp.inference.operators.tensor

import AttributeProto
import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.*
import org.jetbrains.research.kotlin.mpp.inference.operators.*

class Gather(attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int = 1) : Operator<BaseTensor, Tensor>(INFO, usedOutputsNum, attributes) {
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

    override fun apply(inputs: List<BaseTensor>): List<Tensor> {
        val (data, indices) = inputs
        val axis = (data as Tensor).indexAxis((getAttributeValue("axis") as Number).toInt())
        return listOf(data.gather(if (indices is ScalarTensor) indices.toTensor() else indices as Tensor, axis))
    }
}
