package io.kinference.tfjs.operators.layer.normalization

import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.core.fill
import io.kinference.tfjs.externals.core.range
import io.kinference.tfjs.externals.extensions.*
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*

class EmbedLayerNormalization(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(INFO, attributes, inputs, outputs) {

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
            IOInfo(7, setOf(TensorProto.DataType.INT32), "mask", true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", false),
            IOInfo(1, setOf(TensorProto.DataType.INT32), "mask_index", false)
        )

        private val INFO = OperatorInfo("EmbedLayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val epsilon: Float by attribute()

    override fun apply(context: Context, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            val inputIds = inputs[0]!!.data
            val segmentIds = inputs[1]?.data
            val wordWeights = inputs[2]!!.data
            val positionWeights = inputs[3]!!.data
            val segmentWeights = inputs[4]?.data
            val gamma = inputs[5]!!.data
            val beta = inputs[6]!!.data
            val mask = inputs[7]?.data

            val (batchSize, seqLen) = inputIds.shape
            val (_, hiddenSize) = wordWeights.shape

            val outputShape = arrayOf(batchSize, seqLen, hiddenSize)

            val wordEmbedding = wordWeights.gather(inputIds.flatten()).reshape(outputShape)

            val positionIds = range(0, inputIds.shape[1], 1, "int32").broadcastTo(inputIds.shape)

            val positionEmbedding = positionWeights.gather(positionIds.flatten()).reshape(outputShape)


            val segmentEmbedding =
                if (segmentIds != null && segmentWeights != null) {
                    segmentWeights.gather(segmentIds.flatten()).reshape(outputShape)
                } else {
                    null
                }

            val embedding = if (segmentEmbedding != null) {
                wordEmbedding.add(positionEmbedding, segmentEmbedding)
            } else {
                wordEmbedding.plus(positionEmbedding)
            }

            val momentsOutput = embedding.moments(-1, true)
            val mean = momentsOutput.mean
            val variance = momentsOutput.variance

            val epsilonTensor = tensor(floatArrayOf(epsilon), arrayOf(1), "float32")
            val output = (embedding - mean) / (sqrt(variance + epsilonTensor)) * gamma + beta

            val maskOutput = mask?.sum(1, false) ?: fill(arrayOf(batchSize), 0, "int32")
            return@tidy arrayOf(output, maskOutput)
        }

        return listOf(outputs[0].asTensor("output"), outputs[1].asTensor("mask_index"))
    }
}
