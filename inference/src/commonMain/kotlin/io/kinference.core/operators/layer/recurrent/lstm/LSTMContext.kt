package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.core.data.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.graph.ContextPrepare
import io.kinference.ndarray.logger
import io.kinference.core.operators.Operator
import kotlin.time.ExperimentalTime

internal object LSTMContext: ContextPrepare() {
    private val logger = logger("LSTM Initializer")

    @OptIn(ExperimentalTime::class)
    override fun appendContext(context: Context, initializers: List<KITensor>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)
        val peepholesInit = initTensorByDefaultName("P", operator, initializers)

        appendWeights(weightsInit, context)
        appendWeights(recurrentWeightsInit, context)
        appendBias(biasInit, context)
        appendPeepholes(peepholesInit, context)
    }

    internal fun prepareWeights(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 4, shape[1] / 4, shape[2])
        return tensor.data.reshapeView(newShape).toMutable()
               .transpose(intArrayOf(0, 1, 3, 2)).asTensor("prepared_${tensor.info.name}")
    }

    internal fun prepareBias(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 8, shape[1] / 8)
        return tensor.data.toMutable().reshape(newShape).asTensor("prepared_${tensor.info.name}")
    }

    internal fun preparePeepholes(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 3, shape[1] / 3)
        return tensor.data.toMutable().reshape(newShape).asTensor("prepared_${tensor.info.name}")
    }

    private fun appendWeights(tensor: KITensor?, context: Context) {
        if (tensor == null) {
            logger.warn { "Make the weights part of the model, otherwise the LSTM will be slow" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            context.putValue(preparedWeights.info.name, preparedWeights)
        }
    }

    private fun appendBias(tensor: KITensor?, context: Context) {
        if (tensor == null) {
            logger.warn { "Make bias part of the model, otherwise LSTM will be slow" }
        } else {
            val preparedBias = prepareBias(tensor)
            context.putValue(preparedBias.info.name, preparedBias)
        }
    }

    private fun appendPeepholes(tensor: KITensor?, context: Context) {
        if (tensor == null) {
            logger.warn { "Make peepholes part of the model, otherwise LSTM will be slow" }
        } else {
            val preparedPeepholes = preparePeepholes(tensor)
            context.putValue(preparedPeepholes.info.name, preparedPeepholes)
        }
    }
}
