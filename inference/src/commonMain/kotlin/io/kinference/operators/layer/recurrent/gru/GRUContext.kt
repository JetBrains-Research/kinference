package io.kinference.operators.layer.recurrent.gru

import io.kinference.data.ONNXData
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ContextPrepare
import io.kinference.ndarray.logger
import io.kinference.operators.Operator
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
internal object GRUContext: ContextPrepare() {
    private val logger = logger("GRU Initializer")

    override fun appendContext(context: Context, initializers: List<Tensor>, operator: Operator<ONNXData, ONNXData>) {
        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)

        appendWeights(weightsInit, context)
        appendWeights(recurrentWeightsInit, context)
        appendBias(biasInit, context)
    }

    internal fun prepareWeights(tensor: Tensor): Tensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 3, shape[1] / 3, shape[2])
        return tensor.data.reshapeView(newShape).toMutable().transpose(intArrayOf(0, 1, 3, 2)).asTensor("prepared_${tensor.info.name}")
    }

    internal fun prepareBias(tensor: Tensor): Tensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 6, shape[1] / 6)
        return tensor.data.toMutable().reshape(newShape).asTensor("prepared_${tensor.info.name}")
    }

    private fun appendWeights(tensor: Tensor?, context: Context) {
        if (tensor == null) {
            logger.warn { "Make the weights part of the model, otherwise the GRU will be slow" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            context.putValue(preparedWeights.info.name, preparedWeights)
        }
    }

    private fun appendBias(tensor: Tensor?, context: Context) {
        if (tensor == null) {
            logger.warn { "Make the bias part of the model, otherwise the GRU will be slow" }
        } else {
            val preparedBias = prepareBias(tensor)
            context.putValue(preparedBias.info.name, preparedBias)
        }
    }
}
