package org.jetbrains.research.kotlin.mpp.inference.operators.tensor

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class Transpose(attributes: Map<String, Attribute<Any>>) : Operator("Transpose", attributes, emptyList(), INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16,
            TensorProto.DataType.STRING,
            TensorProto.DataType.BOOL,
            TensorProto.DataType.UINT8,
            TensorProto.DataType.COMPLEX128,
            TensorProto.DataType.COMPLEX64,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.INT16,
            TensorProto.DataType.INT8,
            TensorProto.DataType.UINT16
        )

        private val INPUTS_INFO = listOf(InputInfo(0, TYPE_CONSTRAINTS, "data", true)
        )

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "transposed")
        )
    }

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<Tensor> {
        return listOf(inputs.first().transpose(attributes["perm"]?.value as? List<Long>))
    }
}
