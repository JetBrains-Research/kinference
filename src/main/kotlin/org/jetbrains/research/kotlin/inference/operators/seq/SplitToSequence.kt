package org.jetbrains.research.kotlin.inference.operators.seq

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ONNXDataType
import org.jetbrains.research.kotlin.inference.data.seq.TensorSeq
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.tensor.splitWithAxis
import org.jetbrains.research.kotlin.inference.graph.Context
import org.jetbrains.research.kotlin.inference.onnx.AttributeProto
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.AttributeInfo
import org.jetbrains.research.kotlin.inference.operators.IOInfo
import org.jetbrains.research.kotlin.inference.operators.Operator
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo
import org.jetbrains.research.kotlin.inference.types.SequenceInfo

class SplitToSequence(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, TensorSeq>(INFO, attributes, inputs, outputs) {
    companion object {
        private const val DEFAULT_SPLIT_LENGTH = 1
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 0L),
            AttributeInfo("keepdims", setOf(AttributeProto.AttributeType.INT), false, default = 1L)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.INT64, TensorProto.DataType.INT32), "split", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output_sequence", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE))

        private val INFO = OperatorInfo("SplitToSequence", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val axis: Long by attribute()
    private val keepDims: Long by attribute("keepdims")

    override fun apply(context: Context, inputs: List<Tensor?>): List<TensorSeq?> {
        val parts = inputs.elementAtOrNull(1)

        val input = inputs.first()!!
        val tensors = if (parts == null) {
            input.splitWithAxis(input.data.shape[axis.toInt()], axis.toInt(), keepDims == 1L)
        } else {
            input.splitWithAxis(parts, axis.toInt())
        }

        return listOf(TensorSeq(tensors, SequenceInfo("output_sequence", tensors.first().info.type)))
    }
}
