package io.kinference.operators.layer.attention

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.onnx.AttributeProto
import io.kinference.onnx.TensorProto
import io.kinference.operators.*

@ExperimentalUnsignedTypes
class QAttention(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {

    companion object {
        private val FLOATS = setOf(TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16)
        private val BYTES = setOf(TensorProto.DataType.INT8, TensorProto.DataType.UINT8)
        private val DEFAULT_SCALE = FloatNDArray(floatArrayOf(1f))

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("num_heads", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("unidirectional", setOf(AttributeProto.AttributeType.INT), false, default = 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, BYTES, "input", optional = false),
            IOInfo(1, BYTES, "weight", optional = false),
            IOInfo(2, FLOATS, "bias", optional = false),
            IOInfo(3, FLOATS, "input_scale", optional = false),
            IOInfo(4, FLOATS, "weight_scale", optional = false),
            IOInfo(5, setOf(TensorProto.DataType.INT32), "mask_index", optional = true),
            IOInfo(6, BYTES, "input_zero_point", optional = true),
            IOInfo(7, BYTES, "weight_zero_point", optional = true),
            IOInfo(8, FLOATS, "past", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, FLOATS, "output", optional = false),
            IOInfo(1, FLOATS, "present", optional = true)
        )

        private val INFO = OperatorInfo("QAttention", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)

        private fun initQueryKeyValue(input: NDArray, weights: NDArray, bias: NDArray, batchSize: Int, seqLen: Int, hiddenSize: Int, numHeads: Int, outputScale: Float = 1f): Array<MutableNDArray> {
            bias as FloatNDArray
            val qkv = Array(3) { allocateNDArray(input.type, Strides(intArrayOf(batchSize, seqLen, hiddenSize))) }
            val attentionHeadSize = hiddenSize / numHeads

            val step = batchSize * numHeads

            for (i in 0 until 3 * step) {
                val batchIdx = (i / 3) / numHeads
                val headIdx = (i / 3) % numHeads
                val qkvIdx = i % 3

                val inputBatchOffset = batchIdx * seqLen * hiddenSize
                val weightsOffset = qkvIdx * hiddenSize + headIdx * attentionHeadSize
                val qkvOffset = (batchIdx * numHeads + headIdx) * (seqLen * attentionHeadSize)

                //x * W[q|k|v]
                (input as NumberNDArray).gemm(seqLen, attentionHeadSize, hiddenSize, 1.0, hiddenSize, weights as NumberNDArray,
                    3 * hiddenSize, 1.0, qkv[qkvIdx], attentionHeadSize, inputBatchOffset, weightsOffset, qkvOffset)

                repeat(seqLen) {
                    val offset = qkvOffset + it * attentionHeadSize
                    for (j in 0 until attentionHeadSize) {
                        qkv[qkvIdx][j + offset] = qkv[qkvIdx][j + offset] as Float * outputScale + bias[j + weightsOffset]
                    }
                }
            }
            return qkv
        }
    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Boolean by attribute("unidirectional") { it: Number -> it.toInt() == 1 }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val input = inputs[0]!!.data as NumberNDArray
        val weights = inputs[1]!!.data as NumberNDArray

        val qInput = input.dequantize(inputs.getOrNull(6)?.data, DEFAULT_SCALE)
        val qWeight = weights.dequantize(inputs.getOrNull(7)?.data, DEFAULT_SCALE)

        val (batchSize, seqLen, hiddenSize) = input.shape
        val bias = inputs[2]!!.data
        val outputScale = inputs[3]!!.data as FloatNDArray * inputs[4]!!.data as FloatNDArray
        val (queries, keys, values) = initQueryKeyValue(qInput, qWeight, bias, batchSize, seqLen, hiddenSize, numHeads, (outputScale[0] as Float))

        val maskIndices = inputs.elementAtOrNull(5)?.data
        val past = inputs.elementAtOrNull(8)?.data
        val (scores, present) = Attention.getScores(unidir, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize)
        return listOf(scores.asTensor(), present.asTensor())
    }
}