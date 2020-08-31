package io.kinference.operators.layer.normalization

import io.kinference.ndarray.MutableNDArray
import io.kinference.ndarray.NDArray
import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.*
import io.kinference.onnx.AttributeProto
import io.kinference.onnx.TensorProto
import io.kinference.operators.*
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
                var meanSquare = 0.0f

                for (j in 0 until hiddenSize) {
                    val value = this[j + offset] + skip[j + offset] + (bias?.get(j) as? Float ?: 0.0f)
                    dst[j + offset] = value
                    mean += value
                    meanSquare += value * value
                }

                mean /= hiddenSize
                meanSquare = sqrt(meanSquare / hiddenSize - mean * mean + epsilon)

                gamma as FloatNDArray; beta as FloatNDArray
                for (j in 0 until hiddenSize) {
                    dst[j + offset] = (dst[j + offset] - mean) / meanSquare * gamma[j] + beta[j]
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
