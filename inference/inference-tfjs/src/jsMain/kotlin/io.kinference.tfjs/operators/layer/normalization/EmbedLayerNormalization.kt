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

sealed class EmbedLayerNormalization(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
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

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("epsilon", setOf(AttributeProto.AttributeType.FLOAT), false, 0.00001f)
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
        private val INFO = OperatorInfo("EmbedLayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }

    private val epsilon: Float by attribute()

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val inputIds = inputs[0]!!.data as NumberNDArrayTFJS
        val segmentIds = inputs[1]?.data as? NumberNDArrayTFJS
        val wordWeights = inputs[2]!!.data as NumberNDArrayTFJS
        val positionWeights = inputs[3]!!.data as NumberNDArrayTFJS
        val segmentWeights = inputs[4]?.data as? NumberNDArrayTFJS
        val gamma = inputs[5]!!.data as NumberNDArrayTFJS
        val beta = inputs[6]!!.data as NumberNDArrayTFJS
        val mask = inputs[7]?.data as? NumberNDArrayTFJS
        val posIds = inputs[8]?.data as? NumberNDArrayTFJS

        val (batchSize, seqLen) = inputIds.shape
        val (_, hiddenSize) = wordWeights.shape

        val outputShape = intArrayOf(batchSize, seqLen, hiddenSize)
        val outputs = tidyNDArrays {
            val wordEmbedding = wordWeights.gather(inputIds.flatten()).reshape(outputShape) as NumberNDArrayTFJS

            val positionIds = posIds ?: NDArrayTFJS.intRange(start = 0, stop = inputIds.shape[1], step = 1).broadcastTo(inputIds.shapeArray)

            val positionEmbedding = positionWeights.gather(positionIds.flatten()).reshape(outputShape) as NumberNDArrayTFJS

            val segmentEmbedding = if (segmentIds != null && segmentWeights != null) {
                segmentWeights.gather(segmentIds.flatten()).reshape(outputShape) as NumberNDArrayTFJS
            } else {
                null
            }

            val embedding = if (segmentEmbedding != null) {
                wordEmbedding.add(positionEmbedding, segmentEmbedding)
            } else {
                wordEmbedding.plus(positionEmbedding)
            }

            val (mean, variance) = embedding.moments(axis = -1, keepDims = true)

            val epsilonTensor = NDArrayTFJS.float(floatArrayOf(epsilon), arrayOf(1))
            val output = (embedding - mean) / (variance + epsilonTensor).sqrt() * gamma + beta

            val maskOutput = mask?.sum(axis = 1, keepDims = false) ?: NDArrayTFJS.intZeros(arrayOf(batchSize))
            return@tidyNDArrays arrayOf(output, maskOutput, embedding)
        }

        return outputs.asNamedOutputs(this.outputs)
    }
}
