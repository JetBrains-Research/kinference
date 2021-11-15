package io.kinference.core.operators.layer.normalization

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.operators.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.primitives.types.DataType
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.*
import kotlin.math.sqrt

class QEmbedLayerNormalization(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val INT_TYPE = setOf(TensorProto.DataType.INT32)

        private val BYTE_TYPES = setOf(TensorProto.DataType.INT8, TensorProto.DataType.UINT8)

        private val FLOAT_TYPE = setOf(TensorProto.DataType.FLOAT)

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("epsilon", setOf(AttributeProto.AttributeType.FLOAT), true)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, INT_TYPE, "input_ids"),
            IOInfo(1, INT_TYPE, "segment_ids", optional = true),
            IOInfo(2, BYTE_TYPES, "word_embedding_quant"),
            IOInfo(3, BYTE_TYPES, "position_embedding_quant"),
            IOInfo(4, BYTE_TYPES, "segment_embedding", optional = true),
            IOInfo(5, BYTE_TYPES, "gamma_quant"),
            IOInfo(6, BYTE_TYPES, "beta_quant"),
            IOInfo(7, INT_TYPE, "mask", optional = true),
            IOInfo(8, FLOAT_TYPE, "word_embedding_scale"),
            IOInfo(9, FLOAT_TYPE, "position_embedding_scale"),
            IOInfo(10, FLOAT_TYPE, "segment_embedding_scale", optional = true),
            IOInfo(11, FLOAT_TYPE, "gamma_scale"),
            IOInfo(12, FLOAT_TYPE, "beta_scale"),
            IOInfo(13, BYTE_TYPES, "word_embedding_zero_point"),
            IOInfo(14, BYTE_TYPES, "position_embedding_zero_point"),
            IOInfo(15, BYTE_TYPES, "segment_embedding_zero_point", optional = true),
            IOInfo(16, BYTE_TYPES, "gamma_zero_point"),
            IOInfo(17, BYTE_TYPES, "beta_zero_point")
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, FLOAT_TYPE, "layernorm_out", false),
            IOInfo(1, INT_TYPE, "mask_index_out", false)
        )

        private val INFO = OperatorInfo("QEmbedLayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)

        /*private class ByteOrUBytePointer(array: NDArray, startIndex: Int) {
            private val type: TensorProto.DataType
            private val bytePointer: BytePointer?
            private val uBytePointer: UBytePointer?

            init {
                when(array) {
                    is ByteNDArray -> {
                        type = TensorProto.DataType.INT8
                        bytePointer = array.array.pointer(startIndex)
                        uBytePointer = null
                    }
                    is UByteNDArray -> {
                        type = TensorProto.DataType.UINT8
                        bytePointer = null
                        uBytePointer = array.array.pointer(startIndex)
                    }
                    else -> error("Only Byte or UByte")
                }
            }


        }*/
    }

    private val epsilon: Float by attribute()

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val inputIds = inputs[0]!!.data as IntNDArray
        val segmentIds = inputs.getOrNull(1)?.data as IntNDArray?
        val wordEmbedding = inputs[2]!!.data
        val positionEmbedding = inputs[3]!!.data
        val segmentEmbedding = inputs.getOrNull(4)?.data
        val gamma = inputs[5]!!.data
        val beta = inputs[6]!!.data
        val mask = inputs[7]?.data as IntNDArray?
        val wordEmbeddingScale = (inputs[8]!!.data as FloatNDArray).singleValue()
        val positionEmbeddingScale = (inputs[9]!!.data as FloatNDArray).singleValue()
        val segmentEmbeddingScale = (inputs.getOrNull(10)?.data as FloatNDArray?)?.singleValue()
        val gammaScale = (inputs[11]!!.data as FloatNDArray).singleValue()
        val betaScale = (inputs[12]!!.data as FloatNDArray).singleValue()
        val wordEmbeddingZeroPoint = inputs[13]!!.data.singleValue().let {
            when(it) {
                is Byte -> it.toInt()
                is UByte -> it.toInt()
                else -> error("Byte or UByte only")
            }
        }

        val positionEmbeddingZeroPoint = inputs[14]!!.data.singleValue().let {
            when(it) {
                is Byte -> it.toInt()
                is UByte -> it.toInt()
                else -> error("Byte or UByte only")
            }
        }
        val segmentEmbeddingZeroPoint = inputs.getOrNull(15)?.data?.singleValue()?.let {
            when(it) {
                is Byte -> it.toInt()
                is UByte -> it.toInt()
                else -> error("Byte or UByte only")
            }
        }
        val gammaZeroPoint = inputs[16]!!.data.singleValue().let {
            when(it) {
                is Byte -> it.toInt()
                is UByte -> it.toInt()
                else -> error("Byte or UByte only")
            }
        }
        val betaZeroPoint = inputs[17]!!.data.singleValue().let {
            when(it) {
                is Byte -> it.toInt()
                is UByte -> it.toInt()
                else -> error("Byte or UByte only")
            }
        }

        val (batchSize, seqLen) = inputIds.shape
        val (_, hiddenSize) = wordEmbedding.shape
        val output = MutableFloatNDArray(intArrayOf(batchSize, seqLen, hiddenSize))

        val inputType = wordEmbedding.type

        val inputIdsPointer = inputIds.array.pointer()
        val segmentIdsPointer = segmentIds?.array?.pointer()

        repeat(batchSize) { batch ->
            repeat(seqLen) { posIdx ->
                val inputIdx = inputIdsPointer.getAndIncrement()
                val segmentIdx = segmentIdsPointer?.getAndIncrement() ?: 0

                val wordEmbedOffset = inputIdx * hiddenSize
                val segmentEmbedOffset = segmentIdx * hiddenSize
                val posEmbedOffset = posIdx * hiddenSize
                val outputOffset = (posIdx + batch * seqLen) * hiddenSize
                val outputPointer = output.array.pointer(outputOffset)


                var sum = 0.0f
                when(inputType) {
                    DataType.BYTE -> {
                        val wordEmbedPointer = (wordEmbedding as ByteNDArray).array.pointer(wordEmbedOffset)
                        val segmentEmbedPointer = (segmentEmbedding as ByteNDArray?)?.array?.pointer(segmentEmbedOffset)
                        val posEmbedPointer = (positionEmbedding as ByteNDArray).array.pointer(posEmbedOffset)

                        outputPointer.acceptDouble(wordEmbedPointer, posEmbedPointer, hiddenSize) { _: Float, word: Byte, pos: Byte ->
                            val subtotal = (word.toInt() - wordEmbeddingZeroPoint) * wordEmbeddingScale + (pos.toInt() - positionEmbeddingZeroPoint) * positionEmbeddingScale
                            sum += subtotal
                            subtotal
                        }

                        if (segmentEmbedPointer != null) {
                            outputPointer.linearIndex = outputOffset
                            outputPointer.accept(segmentEmbedPointer, hiddenSize) { dst: Float, seg: Byte ->
                                val subtotal = (seg.toInt() - segmentEmbeddingZeroPoint!!) * segmentEmbeddingScale!!
                                sum += subtotal
                                dst + subtotal
                            }
                        }
                    }

                    DataType.UBYTE -> {
                        val wordEmbedPointer = (wordEmbedding as UByteNDArray).array.pointer(wordEmbedOffset)
                        val segmentEmbedPointer = (segmentEmbedding as UByteNDArray?)?.array?.pointer(segmentEmbedOffset)
                        val posEmbedPointer = (positionEmbedding as UByteNDArray).array.pointer(posEmbedOffset)

                        outputPointer.acceptDouble(wordEmbedPointer, posEmbedPointer, hiddenSize) { _: Float, word: UByte, pos: UByte ->
                            val subtotal = (word.toInt() - wordEmbeddingZeroPoint) * wordEmbeddingScale + (pos.toInt() - positionEmbeddingZeroPoint) * positionEmbeddingScale
                            sum += subtotal
                            subtotal
                        }

                        if (segmentEmbedPointer != null) {
                            outputPointer.linearIndex = outputOffset
                            outputPointer.accept(segmentEmbedPointer, hiddenSize) { dst: Float, seg: UByte ->
                                val subtotal = (seg.toInt() - segmentEmbeddingZeroPoint!!) * segmentEmbeddingScale!!
                                sum += subtotal
                                dst + subtotal
                            }
                        }
                    }
                    else -> error("Only Byte or UByte")
                }

                val mean = sum / hiddenSize
                sum = 0.0f
                outputPointer.linearIndex = outputOffset
                outputPointer.map(hiddenSize) { value: Float ->
                    val temp = value - mean
                    sum += temp * temp
                    temp
                }

                val eps = sqrt(sum / hiddenSize + epsilon)
                outputPointer.linearIndex = outputOffset

                when(inputType) {
                    DataType.BYTE -> {
                        val gammaPointer = (gamma as ByteNDArray).array.pointer()
                        val betaPointer = (beta as ByteNDArray).array.pointer()

                        outputPointer.acceptDouble(gammaPointer, betaPointer, hiddenSize) { out: Float, gamma: Byte, beta: Byte ->
                            out / eps * ((gamma.toInt() - gammaZeroPoint) * gammaScale) + ((beta.toInt() - betaZeroPoint) * betaScale)
                        }
                    }

                    DataType.UBYTE -> {
                        val gammaPointer = (gamma as UByteNDArray).array.pointer()
                        val betaPointer = (beta as UByteNDArray).array.pointer()

                        outputPointer.acceptDouble(gammaPointer, betaPointer, hiddenSize) { out: Float, gamma: UByte, beta: UByte ->
                            out / eps * ((gamma.toInt() - gammaZeroPoint) * gammaScale) + ((beta.toInt() - betaZeroPoint) * betaScale)
                        }
                    }
                    else -> error("Only Byte or UByte")
                }
            }
        }

        return listOf(output.asTensor(), EmbedLayerNormalization.createMaskIndices(mask, batchSize, seqLen).asTensor())
    }
}

