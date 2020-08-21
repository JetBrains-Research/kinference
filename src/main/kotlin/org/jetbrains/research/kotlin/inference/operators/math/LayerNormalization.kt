package org.jetbrains.research.kotlin.inference.operators.math

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.DoubleNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.FloatNDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.indexAxis
import org.jetbrains.research.kotlin.inference.graph.Context
import org.jetbrains.research.kotlin.inference.onnx.AttributeProto.AttributeType
import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType
import org.jetbrains.research.kotlin.inference.operators.AttributeInfo
import org.jetbrains.research.kotlin.inference.operators.IOInfo
import org.jetbrains.research.kotlin.inference.operators.Operator
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo
import kotlin.math.sqrt

class LayerNormalization(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    private val axis: Int by attribute { it: Number -> it.toInt() }
    private val epsilon: Float by attribute()

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            DataType.FLOAT,
            DataType.DOUBLE,
            DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeType.INT), false, -1),
            AttributeInfo("epsilon", setOf(AttributeType.FLOAT), false, 0.00001f)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", false),
            IOInfo(1, TYPE_CONSTRAINTS, "scale", false),
            IOInfo(2, TYPE_CONSTRAINTS, "B", false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", false),
            //Unsupported
            IOInfo(1, TYPE_CONSTRAINTS, "mean", true),
            IOInfo(2, TYPE_CONSTRAINTS, "inv_std_var", true)
        )

        private val INFO = OperatorInfo("LayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val input = inputs[0]!!.data
        val scale = inputs[1]!!.data
        val bias = inputs[2]!!.data

        require(input.type == scale.type && input.type == bias.type)
        val type = input.type

        val axis = input.indexAxis(axis)

        var normCount = 1
        var normSize = 1

        for (i in input.shape.indices) {
            if (i < axis) {
                normCount *= input.shape[i]
            } else {
                normSize *= input.shape[i]
            }

        }

        return when (type) {
            DataType.FLOAT -> {
                val inputArray = input.array as FloatArray
                val scaleArray = scale.array as FloatArray
                val biasArray = bias.array as FloatArray

                val outputArray = FloatArray(inputArray.size)
                for (i in 0 until normCount) {
                    val offset = i * normSize

                    var mean = 0.0f
                    var meanSquare = 0.0f
                    for (h in 0 until normSize) {
                        val temp = inputArray[offset + h]
                        mean += temp
                        meanSquare += temp * temp
                    }

                    mean = mean / normSize
                    meanSquare = sqrt(meanSquare / normSize - mean * mean + epsilon)
                    for (h in 0 until normSize) {
                        outputArray[offset + h] = (inputArray[offset + h] - mean) / meanSquare * scaleArray[h] + biasArray[h]
                    }
                }
                listOf(
                    FloatNDArray(outputArray, input.strides).asTensor()
                )
            }
            DataType.DOUBLE -> {
                val inputArray = input.array as DoubleArray
                val scaleArray = scale.array as DoubleArray
                val biasArray = bias.array as DoubleArray

                val outputArray = DoubleArray(inputArray.size)
                for (i in 0 until normCount) {
                    val offset = i * normSize

                    var mean = 0.0
                    var meanSquare = 0.0
                    for (h in 0 until normSize) {
                        val temp = inputArray[offset + h]
                        mean += temp
                        meanSquare += temp * temp
                    }

                    mean = mean / normSize
                    meanSquare = sqrt(meanSquare / normSize - mean * mean + epsilon.toDouble())
                    for (h in 0 until normSize) {
                        outputArray[offset + h] = (inputArray[offset + h] - mean) / meanSquare * scaleArray[h] + biasArray[h]
                    }
                }
                listOf(
                    DoubleNDArray(outputArray, input.strides).asTensor()
                )
            }
            else -> throw UnsupportedOperationException()
        }
    }
}
