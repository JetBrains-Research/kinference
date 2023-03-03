package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.ContextPrepare
import io.kinference.graph.GraphContext
import io.kinference.operator.Operator
import io.kinference.utils.LoggerFactory
import kotlin.time.ExperimentalTime

internal object LSTMContext: ContextPrepare() {
    private val logger = LoggerFactory.create("io.kinference.core.operators.layer.recurrent.lstm.LSTMContext")

    @OptIn(ExperimentalTime::class)
    override suspend fun appendContext(context: GraphContext<KIONNXData<*>>, initializers: List<KITensor>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)
        val peepholesInit = initTensorByDefaultName("P", operator, initializers)

        appendWeights(weightsInit, context)
        appendWeights(recurrentWeightsInit, context)
        appendBias(biasInit, context)
        appendPeepholes(peepholesInit, context)
    }

    internal suspend fun prepareWeights(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 4, shape[1] / 4, shape[2])
        val transposeShape = intArrayOf(0, 1, 3, 2)
        return tensor.data.reshape(newShape).transpose(transposeShape).asTensor("prepared_${tensor.name}")
    }

    internal suspend fun prepareBias(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 8, shape[1] / 8)
        return tensor.data.reshape(newShape).asTensor("prepared_${tensor.name}")
    }

    internal suspend fun preparePeepholes(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 3, shape[1] / 3)
        return tensor.data.reshape(newShape).asTensor("prepared_${tensor.name}")
    }

    private suspend fun appendWeights(tensor: KITensor?, context: GraphContext<KIONNXData<*>>) {
        if (tensor == null) {
            logger.warning { "Make the weights part of the model, otherwise the LSTM will be slow" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            context.putValue(preparedWeights.name!!, preparedWeights)
        }
    }

    private suspend fun appendBias(tensor: KITensor?, context: GraphContext<KIONNXData<*>>) {
        if (tensor == null) {
            logger.warning { "Make bias part of the model, otherwise LSTM will be slow" }
        } else {
            val preparedBias = prepareBias(tensor)
            context.putValue(preparedBias.name!!, preparedBias)
        }
    }

    private suspend fun appendPeepholes(tensor: KITensor?, context: GraphContext<KIONNXData<*>>) {
        if (tensor == null) {
            logger.warning { "Make peepholes part of the model, otherwise LSTM will be slow" }
        } else {
            val preparedPeepholes = preparePeepholes(tensor)
            context.putValue(preparedPeepholes.name!!, preparedPeepholes)
        }
    }
}
