package org.jetbrains.research.kotlin.inference.operators.math

import TensorProto
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.operators.*

class Add(attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int = 1) : Operator<Tensor, Tensor>(INFO, usedOutputsNum, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )

        private val INPUTS_INFO = listOf(
            InputInfo(0, TYPE_CONSTRAINTS, "A", true),
            InputInfo(1, TYPE_CONSTRAINTS, "B", true)
        )

        private val OUTPUTS_INFO = listOf(
            OutputInfo(0, TYPE_CONSTRAINTS, "C")
        )

        private val INFO = OperatorInfo("Add", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(inputs: List<Tensor>): List<Tensor> {
        return listOf(inputs.first() + inputs.last())
    }
}
