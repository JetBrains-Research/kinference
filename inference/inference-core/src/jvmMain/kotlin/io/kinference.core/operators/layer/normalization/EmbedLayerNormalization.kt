package io.kinference.core.operators.layer.normalization

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.*
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.contexts.ManualAllocatorContext
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto.AttributeType
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.PredictionContext
import kotlin.coroutines.coroutineContext
import kotlin.math.sqrt

sealed class EmbedLayerNormalization(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in EmbedLayerNormalizationVer1.VERSION.asRange() -> EmbedLayerNormalizationVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of EmbedLayerNormalization operator: $version")
            }
    }
}


class EmbedLayerNormalizationVer1(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : EmbedLayerNormalization(name, INFO, attributes, inputs, outputs) {

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
            IOInfo(7, setOf(TensorProto.DataType.INT32), "mask", true),
            IOInfo(8, setOf(TensorProto.DataType.INT32), "position_ids", true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", false),
            IOInfo(1, setOf(TensorProto.DataType.INT32), "mask_index", false),
            IOInfo(2, TYPE_CONSTRAINTS, "embedding_sum", true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("EmbedLayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.ORT_DOMAIN)

        private data class NormalizeResult(val output: FloatNDArray, val embeddingSum: FloatNDArray)

        internal suspend fun createMaskIndices(mask: IntNDArray?, batchSize: Int, seqLen: Int, context: ManualAllocatorContext? = null): NumberNDArrayCore {
            val strides = Strides(intArrayOf(batchSize))
            val maskIndices = (context?.getNDArray(DataType.INT, strides) ?: MutableIntNDArray(strides)) as MutableIntNDArray

            if (mask == null)
                return maskIndices.also { it.fill(0) }

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

        private suspend fun normalize(
            epsilon: Float, inputIds: IntNDArray, segmentIds: IntNDArray?, wordEmbed: FloatNDArray, posEmbed: FloatNDArray,
            segmentEmbed: FloatNDArray?, gamma: FloatNDArray, beta: FloatNDArray, positionIds: IntNDArray?, context: ManualAllocatorContext? = null
        ): NormalizeResult {
            val (batchSize, seqLen) = inputIds.shape
            val (_, hiddenSize) = wordEmbed.shape

            val outputStrides = Strides(intArrayOf(batchSize, seqLen, hiddenSize))

            val output = (context?.getNDArray(DataType.FLOAT, outputStrides, fillZeros = false) ?: MutableFloatNDArray(outputStrides)) as MutableFloatNDArray
            val embeddingSum = (context?.getNDArray(DataType.FLOAT, outputStrides, fillZeros = false) ?: MutableFloatNDArray(outputStrides)) as MutableFloatNDArray

            for (batch in 0 until batchSize) {
                val blockIdx = batch * seqLen
                val inputIdsPointer = inputIds.array.pointer(blockIdx)
                val segmentIdsPointer = segmentIds?.array?.pointer(blockIdx)
                val positionIdsPointer = positionIds?.array?.pointer(blockIdx)

                for (seqIdx in 0 until seqLen) {
                    val wordIdx = inputIdsPointer.getAndIncrement()
                    val segmentIdx = segmentIdsPointer?.getAndIncrement() ?: 0
                    val positionIdx = positionIdsPointer?.getAndIncrement() ?: seqIdx

                    val wordEmbedOffset = wordIdx * hiddenSize
                    val segmentEmbedOffset = segmentIdx * hiddenSize
                    val outputOffset = (seqIdx + batch * seqLen) * hiddenSize
                    val posEmbedOffset = positionIdx * hiddenSize

                    val wordEmbedPointer = wordEmbed.array.pointer(wordEmbedOffset)
                    val segmentEmbedPointer = segmentEmbed?.array?.pointer(segmentEmbedOffset)
                    val posEmbedPointer = posEmbed.array.pointer(posEmbedOffset)
                    val embeddingSumPointer = embeddingSum.array.pointer(outputOffset)

                    if (segmentEmbedPointer == null) {
                        embeddingSumPointer.acceptDouble(wordEmbedPointer, posEmbedPointer, hiddenSize) { _, fst, snd ->
                            fst + snd
                        }

                    } else {
                        embeddingSumPointer.acceptTriple(wordEmbedPointer, posEmbedPointer, segmentEmbedPointer, hiddenSize) { _, fst, snd, trd ->
                            fst + snd + trd
                        }
                    }

                    val outputPointer = output.array.pointer(outputOffset)
                    embeddingSumPointer.linearIndex = outputOffset

                    var acc = 0.0f

                    embeddingSumPointer.forEach(hiddenSize) {
                        outputPointer.setAndIncrement(it)
                        acc += it
                    }

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

            return NormalizeResult(output, embeddingSum)
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val manualContext = coroutineContext[ManualAllocatorContext]

        val inputIds = inputs[0]!!.data as IntNDArray
        val segmentIds = inputs[1]?.data as IntNDArray?
        val wordEmbed = inputs[2]!!.data as FloatNDArray
        val posEmbed = inputs[3]!!.data as FloatNDArray
        val segmentEmbed = inputs[4]?.data as FloatNDArray?
        val gamma = inputs[5]!!.data as FloatNDArray
        val beta = inputs[6]!!.data as FloatNDArray
        val mask = inputs.getOrNull(7)?.data as IntNDArray?
        val positionIds = inputs.getOrNull(8)?.data as IntNDArray?

        val (normalized, embedSum) = normalize(epsilon, inputIds, segmentIds, wordEmbed, posEmbed, segmentEmbed, gamma, beta, positionIds, manualContext)
        val maskIndices = createMaskIndices(mask, inputIds.shape[0], inputIds.shape[1])
        return listOf(
            normalized.asTensor(context = manualContext),
            maskIndices.asTensor(context = manualContext),
            embedSum.asTensor(context = manualContext)
        )
    }
}
