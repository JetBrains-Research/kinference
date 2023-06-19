package io.kinference.tfjs.operators.layer.recurrent.gru

import io.kinference.graph.GraphContext
import io.kinference.operator.Operator
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.graph.ContextPrepare
import io.kinference.utils.LoggerFactory

internal object GRUContext : ContextPrepare() {
    private val logger = LoggerFactory.create("io.kinference.tfjs.operators.layer.recurrent.gru.GRUContext")

    override suspend fun appendContext(context: GraphContext<TFJSData<*>>, initializers: List<TFJSTensor>, operator: Operator<TFJSData<*>, TFJSData<*>>) {
        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)

        appendWeights(weightsInit, context)
        appendWeights(recurrentWeightsInit, context)
        appendBias(biasInit, context)
    }

    internal suspend fun prepareWeights(tensor: TFJSTensor): TFJSTensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 3, shape[1] / 3, shape[2])
        return tensor.data.reshape(newShape).transpose(intArrayOf(0, 1, 3, 2)).asTensor("prepared_${tensor.name}")
    }

    internal suspend fun prepareBias(tensor: TFJSTensor): TFJSTensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 6, shape[1] / 6)
        return tensor.data.reshape(newShape).asTensor("prepared_${tensor.name}")
    }

    private suspend fun appendWeights(tensor: TFJSTensor?, context: GraphContext<TFJSData<*>>) {
        if (tensor == null) {
            logger.warning { "Make the weights part of the model, otherwise the GRU will be slow" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            context.putValue(preparedWeights.name!!, preparedWeights)
        }
    }

    private suspend fun appendBias(tensor: TFJSTensor?, context: GraphContext<TFJSData<*>>) {
        if (tensor == null) {
            logger.warning { "Make the bias part of the model, otherwise the GRU will be slow" }
        } else {
            val preparedBias = prepareBias(tensor)
            context.putValue(preparedBias.name!!, preparedBias)
        }
    }
}
