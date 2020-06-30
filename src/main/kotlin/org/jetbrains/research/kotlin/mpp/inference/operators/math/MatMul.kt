package org.jetbrains.research.kotlin.mpp.inference.operators.math

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.InputInfo
import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.operators.OutputInfo
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class MatMul(attributes: Map<String, Attribute<Any>>) : Operator("MatMul", attributes, emptyList(), INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.BFLOAT16
        )

        private val INPUTS_INFO = listOf(
            InputInfo(0, TYPE_CONSTRAINTS, "A", true),
            InputInfo(1, TYPE_CONSTRAINTS, "B", true)
        )

        private val OUTPUTS_INFO = listOf(
            OutputInfo(0, TYPE_CONSTRAINTS, "Y")
        )
    }

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<Tensor> {
        return listOf(inputs.first() matmul inputs.last())
    }
}
