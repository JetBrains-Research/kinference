package io.kinference.operators.quantization.lstm

import io.kinference.data.ONNXData
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ContextPrepare
import io.kinference.ndarray.logger
import io.kinference.operators.Operator
import io.kinference.operators.layer.recurrent.lstm.LSTMContext
import kotlin.time.ExperimentalTime

internal object DynamicQuantizeLSTMContext: ContextPrepare() {
    private val logger = logger("LSTM Initializer")

    @OptIn(ExperimentalTime::class)
    override fun appendContext(context: Context, initializers: List<Tensor>, operator: Operator<ONNXData, ONNXData>) {
        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)
        val peepholesInit = initTensorByDefaultName("P", operator, initializers)

        appendWeights(weightsInit, context)
        appendWeights(recurrentWeightsInit, context)
        appendBias(biasInit, context)
        appendPeepholes(peepholesInit, context)
    }


    internal fun prepareWeights(tensor: Tensor): Tensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], shape[1], 4, shape[2] / 4)
        return tensor.data.toMutable().reshape(newShape)
            .transpose(intArrayOf(0, 2, 1, 3)).asTensor("prepared_${tensor.info.name}")
    }

    private fun appendWeights(tensor: Tensor?, context: Context) {
        if (tensor == null) {
            logger.warn { "Make the weights part of the model, otherwise the LSTM will be slow" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            context.putValue(preparedWeights.info.name, preparedWeights)
        }
    }

    private fun appendBias(tensor: Tensor?, context: Context) {
        if (tensor == null) {
            logger.warn { "Make bias part of the model, otherwise LSTM will be slow" }
        } else {
            val preparedBias = LSTMContext.prepareBias(tensor)
            context.putValue(preparedBias.info.name, preparedBias)
        }
    }

    private fun appendPeepholes(tensor: Tensor?, context: Context) {
        if (tensor == null) {
            logger.warn { "Make peepholes part of the model, otherwise LSTM will be slow" }
        } else {
            val preparedPeepholes = LSTMContext.preparePeepholes(tensor)
            context.putValue(preparedPeepholes.info.name, preparedPeepholes)
        }
    }
}
