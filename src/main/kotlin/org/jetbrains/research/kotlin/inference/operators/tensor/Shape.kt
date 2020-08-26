package org.jetbrains.research.kotlin.inference.operators.tensor

import org.jetbrains.research.kotlin.inference.annotations.DataType
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.data.tensors.asTensor
import org.jetbrains.research.kotlin.inference.graph.Context
import org.jetbrains.research.kotlin.inference.ndarray.extensions.createNDArray
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.IOInfo
import org.jetbrains.research.kotlin.inference.operators.Operator
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo

class Shape(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false))

        private val INFO = OperatorInfo("Shape", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }

    @ExperimentalUnsignedTypes
    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val tensor = inputs.first()!!
        val shape = tensor.data.shape
        val data = LongArray(shape.size) { shape[it].toLong() }
        return listOf(createNDArray(DataType.LONG, data, intArrayOf(data.size)).asTensor("shape"))
    }
}
