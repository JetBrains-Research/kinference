package io.kinference.operators.layer.recurrent.lstm

import io.kinference.data.ONNXData
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.logger
import io.kinference.operators.Operator
import kotlin.time.ExperimentalTime

internal object LSTMContext {
    private val logger = logger("LSTM Initializer")

    @OptIn(ExperimentalTime::class)
    fun appendContext(context: Context, initializers: List<Tensor>, operator: Operator<ONNXData, ONNXData>) {
        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)
        val peepholesInit = initTensorByDefaultName("P", operator, initializers)

        appendWeights(weightsInit, context)
        appendWeights(recurrentWeightsInit, context)
        appendBias(biasInit, context)
        appendPeepholes(peepholesInit, context)
    }

    @OptIn(ExperimentalTime::class)
    private fun initTensorByDefaultName(defaultName: String, operator: Operator<ONNXData, ONNXData>, initializers: List<Tensor>): Tensor? {
        val index = operator.info.inputs.find { it.name == defaultName }?.index ?: return null
        val tensorName = operator.inputs.getOrNull(index)

        return initializers.find { it.info.name == tensorName }
    }


    internal fun prepareWeights(tensor: Tensor): Tensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 4, shape[1] / 4, shape[2])
        return tensor.data.reshapeView(newShape).toMutable()
               .transpose(intArrayOf(0, 1, 3, 2)).asTensor("prepared_${tensor.info.name}")
    }

    internal fun prepareBias(tensor: Tensor): Tensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 8, shape[1] / 8)
        return tensor.data.toMutable().reshape(newShape).asTensor("prepared_${tensor.info.name}")
    }

    internal fun preparePeepholes(tensor: Tensor): Tensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 3, shape[1] / 3)
        return tensor.data.toMutable().reshape(newShape).asTensor("prepared_${tensor.info.name}")
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
            val preparedBias = prepareBias(tensor)
            context.putValue(preparedBias.info.name, preparedBias)
        }
    }

    private fun appendPeepholes(tensor: Tensor?, context: Context) {
        if (tensor == null) {
            logger.warn { "Make peepholes part of the model, otherwise LSTM will be slow" }
        } else {
            val preparedPeepholes = preparePeepholes(tensor)
            context.putValue(preparedPeepholes.info.name, preparedPeepholes)
        }
    }
}
