package io.kinference.core.operators.layer.recurrent.gru

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.*
import io.kinference.graph.GraphContext
import io.kinference.operator.Operator
import io.kinference.utils.LoggerFactory
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
internal object GRUContext: ContextPrepare() {
    private val logger = LoggerFactory.create("io.kinference.core.operators.layer.recurrent.gru.GRUContext")

    override suspend fun appendContext(context: GraphContext<KIONNXData<*>>, initializers: List<KITensor>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)

        appendWeights(weightsInit, context)
        appendWeights(recurrentWeightsInit, context)
        appendBias(biasInit, context)
    }

    internal suspend fun prepareWeights(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 3, shape[1] / 3, shape[2])
        return tensor.data.reshape(newShape).transpose(intArrayOf(0, 1, 3, 2)).asTensor("prepared_${tensor.name}")
    }

    internal suspend fun prepareBias(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 6, shape[1] / 6)
        return tensor.data.reshape(newShape).asTensor("prepared_${tensor.name}")
    }

    private suspend fun appendWeights(tensor: KITensor?, context: GraphContext<KIONNXData<*>>) {
        if (tensor == null) {
            logger.warning { "Make the weights part of the model, otherwise the GRU will be slow" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            context.putValue(preparedWeights.name!!, preparedWeights)
        }
    }

    private suspend fun appendBias(tensor: KITensor?, context: GraphContext<KIONNXData<*>>) {
        if (tensor == null) {
            logger.warning { "Make the bias part of the model, otherwise the GRU will be slow" }
        } else {
            val preparedBias = prepareBias(tensor)
            context.putValue(preparedBias.name!!, preparedBias)
        }
    }
}
