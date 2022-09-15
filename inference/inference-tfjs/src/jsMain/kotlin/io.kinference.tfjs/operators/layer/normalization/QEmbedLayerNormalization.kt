package io.kinference.tfjs.operators.layer.normalization

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asNamedOutputs

sealed class QEmbedLayerNormalization(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in QEmbedLayerNormalizationVer1.VERSION.asRange() -> QEmbedLayerNormalizationVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of QEmbedLayerNormalization operator: $version")
            }
    }
}

class QEmbedLayerNormalizationVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    QEmbedLayerNormalization(name, INFO, attributes, inputs, outputs) {
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

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("QEmbedLayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, "com.microsoft")
    }

    private val epsilon: Float by attribute()

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val inputIds = inputs[0]!!.data
        val segmentIds = inputs.getOrNull(1)?.data
        val wordEmbedding = inputs[2]!!.data as NumberNDArrayTFJS
        val positionEmbedding = inputs[3]!!.data as NumberNDArrayTFJS
        val segmentEmbedding = inputs.getOrNull(4)?.data as? NumberNDArrayTFJS
        val gamma = inputs[5]!!.data as NumberNDArrayTFJS
        val beta = inputs[6]!!.data as NumberNDArrayTFJS
        val mask = inputs[7]?.data as? NumberNDArrayTFJS
        val wordEmbeddingScale = inputs[8]!!.data as NumberNDArrayTFJS
        val positionEmbeddingScale = inputs[9]!!.data as NumberNDArrayTFJS
        val segmentEmbeddingScale = inputs.getOrNull(10)?.data as? NumberNDArrayTFJS
        val gammaScale = inputs[11]!!.data as NumberNDArrayTFJS
        val betaScale = inputs[12]!!.data as NumberNDArrayTFJS
        val wordEmbeddingZeroPoint = inputs[13]!!.data as NumberNDArrayTFJS
        val positionEmbeddingZeroPoint = inputs[14]!!.data as NumberNDArrayTFJS
        val segmentEmbeddingZeroPoint = inputs[15]?.data as? NumberNDArrayTFJS
        val gammaZeroPoint = inputs[16]!!.data as NumberNDArrayTFJS
        val betaZeroPoint = inputs[17]!!.data as NumberNDArrayTFJS

        val (batchSize, seqLen) = inputIds.shape
        val (_, hiddenSize) = wordEmbedding.shape

        val outputShape = intArrayOf(batchSize, seqLen, hiddenSize)
        val outputs = tidyNDArrays {
            val dequantWordEmbedding = (wordEmbedding - wordEmbeddingZeroPoint) * wordEmbeddingScale
            val dequantPositionEmbedding = (positionEmbedding - positionEmbeddingZeroPoint) * positionEmbeddingScale
            val dequantSegmentEmbedding = if (segmentEmbedding != null && segmentEmbeddingZeroPoint != null && segmentEmbeddingScale != null) {
                (segmentEmbedding - segmentEmbeddingZeroPoint) * segmentEmbeddingScale
            } else {
                null
            }

            val wordResult = dequantWordEmbedding.gather(inputIds.flatten()).reshape(outputShape) as NumberNDArrayTFJS

            val positionIds = NumberNDArrayTFJS(range(0, inputIds.shape[1], 1, "int32")).broadcastTo(inputIds.shapeArray)
            val positionResult = dequantPositionEmbedding.gather(positionIds.flatten()).reshape(outputShape) as NumberNDArrayTFJS

            val segmentResult = if (dequantSegmentEmbedding != null && segmentIds != null) {
                dequantSegmentEmbedding.gather(segmentIds.flatten()).reshape(outputShape) as NumberNDArrayTFJS
            } else {
                null
            }

            val result = if (segmentResult != null) {
                wordResult.add(positionResult, segmentResult)
            } else {
                wordResult + positionResult
            }

            val (mean, variance) = result.moments(axis = -1, keepDims = true)

            val epsilonTensor = NumberNDArrayTFJS(scalar(epsilon))
            val dequantGamma = (gamma - gammaZeroPoint) * gammaScale
            val dequantBeta = (beta - betaZeroPoint) * betaScale
            val output = (result - mean) / (variance + epsilonTensor).sqrt() * dequantGamma + dequantBeta

            val maskOutput = mask?.sum(1, false) ?: NumberNDArrayTFJS(fill(arrayOf(batchSize), 0, "int32"))

            return@tidyNDArrays arrayOf(output, maskOutput)
        }
        return outputs.asNamedOutputs(this.outputs)
    }
}
