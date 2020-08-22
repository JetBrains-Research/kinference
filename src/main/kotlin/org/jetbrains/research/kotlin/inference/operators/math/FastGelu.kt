package org.jetbrains.research.kotlin.inference.operators.math

import org.jetbrains.research.kotlin.inference.annotations.DataType
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.graph.Context
import org.jetbrains.research.kotlin.inference.math.*
import org.jetbrains.research.kotlin.inference.operators.IOInfo
import org.jetbrains.research.kotlin.inference.operators.Operator
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo

class FastGelu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) :
    Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "bias", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false)
        )

        private val INFO = OperatorInfo("FastGelu", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val input = inputs.first()!!
        val bias = inputs.getOrNull(1)

        val result = when (input.data.type) {
            DataType.FLOAT -> {
                val biasData = bias?.data as? FloatNDArray
                val result = input.data.toMutable() as MutableFloatNDArray
                if (bias?.data == null) {
                    for (i in 0 until result.linearSize) result[i] = fgelu(result[i])
                } else {
                    for (i in 0 until result.linearSize) result[i] = fgelu(result[i] + biasData!![i % biasData.linearSize])
                }
                result
                /*override fun apply(array: FloatArray): FloatArray {
                    if (bias == null) {
                        for (i in array.indices) array[i] = fgelu(array[i])
                    } else {
                        val biasArray = bias.data.array as FloatArray
//                    require(biasArray.size == array.size) { "FastGelu: Bias length must be same as input" }
                        for (i in array.indices) array[i] = fgelu(array[i] + biasArray[i % biasArray.size])
                    }

                    return array
                }*/
            }

            DataType.DOUBLE -> {
                val biasData = bias?.data as? DoubleNDArray
                val result = input.data.toMutable() as MutableDoubleNDArray
                if (bias?.data == null) {
                    for (i in 0 until result.linearSize) result[i] = fgelu(result[i])
                } else {
                    for (i in 0 until result.linearSize) result[i] = fgelu(result[i] + biasData!![i % biasData.linearSize])
                }
                /*input.data.mapElements(object : DoubleArrayToDoubleArray {
                override fun apply(array: DoubleArray): DoubleArray {
                    if (bias == null) {
                        for (i in array.indices) array[i] = fgelu(array[i])
                    } else {
                        val biasArray = bias.data.array as DoubleArray
//                    require(biasArray.size == array.size) { "FastGelu: Bias length must be same as input" }
                        for (i in array.indices) array[i] = fgelu(array[i] + biasArray[i % biasArray.size])
                    }

                    return array
                }*/
                result
            }

            else -> error("Unsupported operation")
        }.asTensor("Y")

        return listOf(result)
    }
}
