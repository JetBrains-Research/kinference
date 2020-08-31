package io.kinference.operators.seq

import io.kinference.attributes.Attribute
import io.kinference.data.ONNXDataType
import io.kinference.data.seq.TensorSeq
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.concatenate
import io.kinference.data.tensors.stack
import io.kinference.graph.Context
import io.kinference.onnx.AttributeProto
import io.kinference.operators.AttributeInfo
import io.kinference.operators.IOInfo
import io.kinference.operators.Operator
import io.kinference.operators.OperatorInfo

class ConcatFromSequence(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<TensorSeq, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("new_axis", setOf(AttributeProto.AttributeType.INT), false, default = 0L)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input_sequence", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "concat_result", optional = false))

        private val INFO = OperatorInfo("ConcatFromSequence", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val axis: Long by attribute()
    private val newAxis: Long by attribute("new_axis")

    override fun apply(context: Context, inputs: List<TensorSeq?>): List<Tensor?> {
        val srcTensors = inputs.first()!!.data
        val tensor = if (newAxis == 1L) srcTensors.stack(axis.toInt()) else srcTensors.concatenate(axis.toInt())
        return listOf(tensor)
    }
}
