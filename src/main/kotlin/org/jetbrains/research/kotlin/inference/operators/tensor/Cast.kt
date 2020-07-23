package org.jetbrains.research.kotlin.inference.operators.tensor

import AttributeProto
import TensorProto
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.*
import org.jetbrains.research.kotlin.inference.operators.*

class Cast(attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int = 1)
    : Operator<BaseTensor, BaseTensor>(INFO, usedOutputsNum, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("to", setOf(AttributeProto.AttributeType.INT), true)
        )

        private val INPUTS_INFO = listOf(InputInfo(0, TYPE_CONSTRAINTS, "input"))

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))

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

    override fun apply(inputs: List<BaseTensor>): List<BaseTensor> {
        val tensor = inputs.first()
        val to = TensorProto.DataType.fromValue(getAttributeValue("to") as Int)!!
        return when (tensor) {
            is ScalarTensor -> listOf(ScalarTensor(tensor.info.name, cast(tensor.value, tensor.info.type, to), to))
            is Tensor -> listOf(tensor.mapElements(to) { cast(it, tensor.info.type, to) })
            else -> error("Unsupported operation")
        }
    }
}
