package io.kinference.operators.tensor

import io.kinference.primitives.types.DataType
import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.extensions.createNDArray
import io.kinference.ndarray.arrays.tiled.LongTiledArray
import io.kinference.onnx.TensorProto
import io.kinference.operators.IOInfo
import io.kinference.operators.Operator
import io.kinference.operators.OperatorInfo

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

        val outputTensorShape = intArrayOf(shape.size)
        val data = LongTiledArray(outputTensorShape) { shape[it].toLong() }
        return listOf(createNDArray(DataType.LONG, data, outputTensorShape).asTensor("shape"))
    }
}
