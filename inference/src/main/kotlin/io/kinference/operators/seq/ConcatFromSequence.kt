package io.kinference.operators.seq

import io.kinference.attributes.Attribute
import io.kinference.data.ONNXDataType
import io.kinference.data.seq.TensorSeq
import io.kinference.data.tensors.*
import io.kinference.graph.Context
import io.kinference.onnx.AttributeProto
import io.kinference.operators.*

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

    private val axis: Int by attribute { it: Number -> it.toInt() }
    private val newAxis: Boolean by attribute("new_axis") { it: Number -> it.toInt() == 1 }

    override fun apply(context: Context, inputs: List<TensorSeq?>): List<Tensor?> {
        val srcTensors = inputs.first()!!.data
        val tensor = if (newAxis) srcTensors.stack(axis) else srcTensors.concatenate(axis)
        return listOf(tensor)
    }
}
