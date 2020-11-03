package io.kinference.operators.layer.normalization

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.NDArray
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.onnx.AttributeProto.AttributeType
import io.kinference.onnx.TensorProto
import io.kinference.operators.*
import kotlin.math.sqrt

@ExperimentalUnsignedTypes
class EmbedLayerNormalization(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    private val epsilon: Float by attribute()

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("epsilon", setOf(AttributeType.FLOAT), false, 0.00001f)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.INT32), "input_ids", false),
            IOInfo(1, setOf(TensorProto.DataType.INT32), "segment_ids", true),
            IOInfo(2, TYPE_CONSTRAINTS, "word_embedding", false),
            IOInfo(3, TYPE_CONSTRAINTS, "position_embedding", false),
            IOInfo(4, TYPE_CONSTRAINTS, "segment_embedding", true),
            IOInfo(5, TYPE_CONSTRAINTS, "gamma", false),
            IOInfo(6, TYPE_CONSTRAINTS, "beta", false),
            IOInfo(7, setOf(TensorProto.DataType.INT32), "mask", true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", false),
            IOInfo(1, setOf(TensorProto.DataType.INT32), "mask_index", false)
        )

        private val INFO = OperatorInfo("EmbedLayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)

        fun createMaskIndices(mask: NDArray?, batchSize: Int, seqLen: Int): NumberNDArray {
            val maskIndices = MutableIntNDArray(IntArray(batchSize), Strides(intArrayOf(batchSize)))
            if (mask == null) return maskIndices

            for (i in 0 until batchSize) {
                val offset = i * seqLen
                var count = 0
                for (j in offset until offset + seqLen) if ((mask[j] as Number).toInt() == 1) count += 1
                maskIndices[i] = count
            }
            return maskIndices
        }

        fun normalize(epsilon: Float, inputIds: NDArray, segmentIds: NDArray?, wordEmbed: NDArray, posEmbed: NDArray, segmentEmbed: NDArray?, gamma: NDArray, beta: NDArray): MutableFloatNDArray {
            val (batchSize, seqLen) = inputIds.shape
            val (_, hiddenSize) = wordEmbed.shape
            val output = MutableFloatNDArray(FloatArray(batchSize * seqLen * hiddenSize), Strides(intArrayOf(batchSize, seqLen, hiddenSize)))

            val steps = seqLen * batchSize
            repeat(steps) { i ->
                val wordIdx = (inputIds[i] as Number).toInt()
                val posIdx = i % seqLen
                val segmentIdx = (segmentIds?.get(i) as? Number)?.toInt() ?: 0

                val outputOffset = i * hiddenSize
                val wordEmbedOffset = wordIdx * hiddenSize
                val posEmbedOffset = posIdx * hiddenSize
                val segmentEmbedOffset = segmentIdx * hiddenSize

                //only floats supported
                wordEmbed as FloatNDArray; posEmbed as FloatNDArray
                var acc = 0.0f
                for (j in 0 until hiddenSize) {
                    var tmp = wordEmbed[j + wordEmbedOffset] + posEmbed[j + posEmbedOffset];
                    if (segmentEmbed != null)
                        tmp += segmentEmbed[j + segmentEmbedOffset] as Float
                    output[j + outputOffset] = tmp
                    acc += tmp
                }
                val mean = acc / hiddenSize
                acc = 0.0f
                for (j in 0 until hiddenSize) {
                    output[j + outputOffset] -= mean
                    acc += output[j + outputOffset] * output[j + outputOffset]
                }

                gamma as FloatNDArray; beta as FloatNDArray
                val eps = sqrt(acc / hiddenSize + epsilon)
                for (j in 0 until hiddenSize) {
                    output[j + outputOffset] = output[j + outputOffset] / eps * gamma[j] + beta[j]
                }
            }
            return output
        }
    }

    @ExperimentalUnsignedTypes
    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val inputIds = inputs[0]!!.data
        val segmentIds = inputs[1]?.data
        val wordEmbed = inputs[2]!!.data
        val posEmbed = inputs[3]!!.data
        val segmentEmbed = inputs[4]?.data
        val gamma = inputs[5]!!.data
        val beta = inputs[6]!!.data
        val mask = inputs.getOrNull(7)?.data

        val normalized = normalize(epsilon, inputIds, segmentIds, wordEmbed, posEmbed, segmentEmbed, gamma, beta).asTensor("output")
        val maskIndices = createMaskIndices(mask, inputIds.shape[0], inputIds.shape[1]).asTensor("mask_index")
        return listOf(normalized, maskIndices)
    }
}
