package org.jetbrains.research.kotlin.mpp.inference.operators.math

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor

class Add(attributes: Map<String, Attribute<Any>>) : Operator<Tensor, Tensor>("Add", attributes, emptyList(), INPUTS_INFO, OUTPUTS_INFO) {
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
    }

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<Tensor> {
        return listOf(inputs.first() + inputs.last())
    }
}
