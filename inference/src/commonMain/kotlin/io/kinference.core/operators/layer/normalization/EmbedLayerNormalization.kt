package io.kinference.core.operators.layer.normalization

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.graph.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.core.operators.*
import io.kinference.protobuf.message.AttributeProto.AttributeType
import io.kinference.protobuf.message.TensorProto
import kotlin.math.sqrt
import kotlin.time.ExperimentalTime

@ExperimentalTime
class EmbedLayerNormalization(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
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

        fun createMaskIndices(mask: IntNDArray?, batchSize: Int, seqLen: Int): NumberNDArray {
            val maskIndices = MutableIntNDArray(shape = intArrayOf(batchSize))
            if (mask == null) return maskIndices

            val pointer = mask.array.pointer()
            val maskIndicesPointer = maskIndices.array.pointer()
            for (i in 0 until batchSize) {
                var count = 0
                pointer.linearIndex = i * seqLen
                pointer.forEach(seqLen) {
                    if (it == 1) count += 1
                }

                maskIndicesPointer.set(count)
                maskIndicesPointer.increment()
            }

            return maskIndices
        }

        fun normalize(epsilon: Float, inputIds: IntNDArray, segmentIds: IntNDArray?,
                      wordEmbed: FloatNDArray, posEmbed: FloatNDArray, segmentEmbed: FloatNDArray?, gamma: FloatNDArray, beta: FloatNDArray): MutableFloatNDArray {
            val (batchSize, seqLen) = inputIds.shape
            val (_, hiddenSize) = wordEmbed.shape
            val output = MutableFloatNDArray(shape = intArrayOf(batchSize, seqLen, hiddenSize))

            val inputIdsPointer = inputIds.array.pointer()
            val segmentIdsPointer = segmentIds?.array?.pointer()
            for (batch in 0 until batchSize) {
                for (posIdx in 0 until seqLen) {
                    val wordIdx = inputIdsPointer.getAndIncrement()
                    val segmentIdx = segmentIdsPointer?.getAndIncrement() ?: 0

                    val wordEmbedOffset = wordIdx * hiddenSize
                    val segmentEmbedOffset = segmentIdx * hiddenSize
                    val outputOffset = (posIdx + batch * seqLen) * hiddenSize
                    val posEmbedOffset = posIdx * hiddenSize

                    val wordEmbedPointer = wordEmbed.array.pointer(wordEmbedOffset)
                    val segmentEmbedPointer = segmentEmbed?.array?.pointer(segmentEmbedOffset)
                    val outputPointer = output.array.pointer(outputOffset)
                    val posEmbedPointer = posEmbed.array.pointer(posEmbedOffset)

                    if (segmentEmbedPointer == null) {
                        outputPointer.acceptDouble(wordEmbedPointer, posEmbedPointer, hiddenSize) { _, fst, snd ->
                            fst + snd
                        }
                    } else {
                        outputPointer.acceptTriple(wordEmbedPointer, posEmbedPointer, segmentEmbedPointer, hiddenSize) { _, fst, snd, trd ->
                            fst + snd + trd
                        }
                    }

                    var acc = 0.0f
                    outputPointer.linearIndex = outputOffset
                    outputPointer.forEach(hiddenSize) { acc += it }

                    val mean = acc / hiddenSize
                    acc = 0.0f

                    outputPointer.linearIndex = outputOffset
                    outputPointer.map(hiddenSize) { it - mean }

                    outputPointer.linearIndex = outputOffset
                    outputPointer.forEach(hiddenSize) { acc += it * it }

                    val eps = sqrt(acc / hiddenSize + epsilon)
                    outputPointer.linearIndex = outputOffset
                    val gammaPointer = gamma.array.pointer()
                    val betaPointer = beta.array.pointer()

                    outputPointer.acceptDouble(gammaPointer, betaPointer, hiddenSize) { out, g, b -> out / eps * g + b }
                }
            }

            return output
        }
    }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val inputIds = inputs[0]!!.data as IntNDArray
        val segmentIds = inputs[1]?.data as IntNDArray?
        val wordEmbed = inputs[2]!!.data as FloatNDArray
        val posEmbed = inputs[3]!!.data as FloatNDArray
        val segmentEmbed = inputs[4]?.data as FloatNDArray?
        val gamma = inputs[5]!!.data as FloatNDArray
        val beta = inputs[6]!!.data as FloatNDArray
        val mask = inputs.getOrNull(7)?.data as IntNDArray?

        val normalized = normalize(epsilon, inputIds, segmentIds, wordEmbed, posEmbed, segmentEmbed, gamma, beta).asTensor("output")
        val maskIndices = createMaskIndices(mask, inputIds.shape[0], inputIds.shape[1]).asTensor("mask_index")
        return listOf(normalized, maskIndices)
    }
}
