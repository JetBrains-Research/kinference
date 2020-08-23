package org.jetbrains.research.kotlin.inference.operators.tensor

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.graph.Context
import org.jetbrains.research.kotlin.inference.onnx.AttributeProto
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.AttributeInfo
import org.jetbrains.research.kotlin.inference.operators.IOInfo
import org.jetbrains.research.kotlin.inference.operators.Operator
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo

class Cast(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("to", setOf(AttributeProto.AttributeType.INT), true)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val INFO = OperatorInfo("Cast", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private fun cast(value: Any, from: TensorProto.DataType, to: TensorProto.DataType): Any {
        val base: Number = when (from) {
            TensorProto.DataType.BOOL -> if (value as Boolean) 1 else 0
            TensorProto.DataType.INT64, TensorProto.DataType.INT32, TensorProto.DataType.FLOAT, TensorProto.DataType.DOUBLE -> value as Number
            else -> error("Unsupported operation")
        }

        @Suppress("IMPLICIT_CAST_TO_ANY") val casted = when (to) {
            TensorProto.DataType.DOUBLE -> base.toDouble()
            TensorProto.DataType.FLOAT -> base.toFloat()
            TensorProto.DataType.INT64 -> base.toLong()
            TensorProto.DataType.INT32 -> base.toInt()
            TensorProto.DataType.BOOL -> base != 0
            else -> error("Unsupported operation")
        }

        return casted
    }

    private val toType: Int by attribute("to") { it: Number -> it.toInt() }

    @ExperimentalUnsignedTypes
    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val tensor = inputs.first()!!
        val to = TensorProto.DataType.fromValue(toType)!!
        return listOf(tensor.mapElements(to) { cast(it, tensor.info.type, to) })
    }
}
