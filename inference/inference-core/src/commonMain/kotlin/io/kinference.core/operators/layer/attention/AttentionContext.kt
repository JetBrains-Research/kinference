package io.kinference.core.operators.layer.attention

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.*
import io.kinference.graph.GraphContext
import io.kinference.operator.Operator
import io.kinference.utils.LoggerFactory
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
internal object AttentionContext: ContextPrepare() {
    private val logger = LoggerFactory.create("io.kinference.core.operators.layer.attention.AttentionContext")

    override fun appendContext(context: GraphContext<KIONNXData<*>>, initializers: List<KITensor>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        val weightsInit = initTensorByDefaultName("weight", operator, initializers)
        val biasInit = initTensorByDefaultName("bias", operator, initializers)
        val numHeads = operator.getAttribute<Long>("num_heads").toInt()

        appendWeights(weightsInit, context, numHeads)
        appendBias(biasInit, context, numHeads)
    }

    internal fun prepareWeights(tensor: KITensor, numHeads: Int): KITensor {
        val shape = tensor.data.shape
        val headSize = shape[1] / 3 / numHeads
        val newShape = intArrayOf(shape[0], 3, numHeads, headSize)

        val prepared = tensor.data.reshape(newShape).transpose(intArrayOf(1, 2, 0, 3))

        return prepared.asTensor("prepared_${tensor.name}")
    }

    internal fun prepareBias(tensor: KITensor, numHeads: Int): KITensor {
        val shape = tensor.data.shape
        val headSize = shape[0] / 3 / numHeads
        val newShape = intArrayOf(3, numHeads, headSize)
        return tensor.data.reshape(newShape).asTensor("prepared_${tensor.name}")
    }

    private fun appendWeights(tensor: KITensor?, context: GraphContext<KIONNXData<*>>, numHeads: Int) {
        if (tensor == null) {
            logger.warning { "Make the weights part of the model, otherwise the Attention will be slow" }
        } else {
            val preparedWeights = prepareWeights(tensor, numHeads)
            context.putValue(preparedWeights.name!!, preparedWeights)
        }
    }

    private fun appendBias(tensor: KITensor?, context: GraphContext<KIONNXData<*>>, numHeads: Int) {
        if (tensor == null) {
            logger.warning { "Make the bias part of the model, otherwise the Attention will be slow\n" }
        } else {
            val preparedBias = prepareBias(tensor, numHeads)
            context.putValue(preparedBias.name!!, preparedBias)
        }
    }
}
