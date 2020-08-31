package org.jetbrains.research.kotlin.inference.operators.layer.normalization

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.data.tensors.asTensor
import org.jetbrains.research.kotlin.inference.graph.Context
import org.jetbrains.research.kotlin.inference.ndarray.*
import org.jetbrains.research.kotlin.inference.onnx.AttributeProto
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.*
import kotlin.math.sqrt

class SkipLayerNormalization(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    private val epsilon: Float by attribute()

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("epsilon", setOf(AttributeProto.AttributeType.FLOAT), false, 0.00001f)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input", false),
            IOInfo(1, TYPE_CONSTRAINTS, "skip", false),
            IOInfo(2, TYPE_CONSTRAINTS, "gamma", false),
            IOInfo(3, TYPE_CONSTRAINTS, "beta", false),
            IOInfo(4, TYPE_CONSTRAINTS, "bias", true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", false),
            //Only for training, not supported now
            IOInfo(1, TYPE_CONSTRAINTS, "mean", true),
            IOInfo(2, TYPE_CONSTRAINTS, "inv_std_var", true)
        )

        private val INFO = OperatorInfo("SkipLayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)

        @ExperimentalUnsignedTypes
        private fun NDArray.normalize(skip: NDArray, gamma: NDArray, beta: NDArray, bias: NDArray?, epsilon: Float, dst: MutableNDArray) {
            val (batchSize, seqLen, hiddenSize) = this.shape
            val steps = batchSize * seqLen

            //only floats supported
            this as FloatNDArray; skip as FloatNDArray; dst as MutableFloatNDArray
            for (i in 0 until steps) {
                val offset = hiddenSize * i

                var mean = 0.0f
                var meanSqrt = 0.0f

                for (j in offset until hiddenSize + offset) {
                    val value = this[j] + skip[j] + (bias?.get(j) as? Float ?: 0.0f)
                    dst[j] = value
                    mean += value
                    meanSqrt = sqrt(meanSqrt / hiddenSize - mean * mean + epsilon)
                }

                gamma as FloatNDArray; beta as FloatNDArray
                for (j in 0 until hiddenSize) {
                    dst[j] = (dst[j] - mean) / meanSqrt * gamma[j] + beta[j]
                }
            }
        }
    }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val input = inputs[0]!!.data
        val output = input.allocateNDArray(input.strides)
        input.normalize(
            skip = inputs[1]!!.data,
            gamma = inputs[2]!!.data,
            beta = inputs[3]!!.data,
            bias = inputs.getOrNull(4)?.data,
            epsilon = epsilon,
            dst = output
        )
        return listOf(output.asTensor())
    }
}
