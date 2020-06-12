package org.jetbrains.research.kotlin.mpp.inference.operators.tensor

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.tensors.*

class Split(attributes: Map<String, Attribute<Any>>) : Operator("Split", attributes, emptyList(), INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT64,
            TensorProto.DataType.UINT16,
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
            TensorProto.DataType.INT8
        )

        private val INPUTS_INFO = listOf(
            InputInfo(0, TYPE_CONSTRAINTS, "input", true),
            InputInfo(1, setOf(TensorProto.DataType.INT64), "split", false)
        )

        private val OUTPUTS_INFO = listOf(
            OutputInfo(0, TYPE_CONSTRAINTS, "output")
        )
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
