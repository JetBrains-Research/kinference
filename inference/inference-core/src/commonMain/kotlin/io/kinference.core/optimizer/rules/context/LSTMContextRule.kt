package io.kinference.core.optimizer.rules.context

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.operators.layer.recurrent.lstm.LSTM
import io.kinference.graph.Graph
import io.kinference.operator.Operator
import io.kinference.utils.LoggerFactory

object LSTMContextRule : PrepareContextRule(operatorName = "LSTM") {
    private val logger = LoggerFactory.create("io.kinference.core.optimizer.rules.context.LSTMContextRule")

    internal suspend fun prepareWeights(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 4, shape[1] / 4, shape[2])
        val transposeShape = intArrayOf(0, 1, 3, 2)
        return tensor.data.reshape(newShape).transpose(transposeShape).asTensor("${PREFIX}_${tensor.name}")
    }

    internal suspend fun prepareBias(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 8, shape[1] / 8)
        return tensor.data.reshape(newShape).asTensor("${PREFIX}_${tensor.name}")
    }

    internal suspend fun preparePeepholes(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 3, shape[1] / 3)
        return tensor.data.reshape(newShape).asTensor("${PREFIX}_${tensor.name}")
    }

    private suspend fun appendWeights(tensor: KITensor?, graph: KIGraph) {
        if (tensor == null) {
            logger.warning { "Make the weights part of the model, otherwise the LSTM will be slow" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            graph.addTensorToContext(preparedWeights)
        }
    }

    private suspend fun appendBias(tensor: KITensor?, graph: KIGraph) {
        if (tensor == null) {
            logger.warning { "Make bias part of the model, otherwise LSTM will be slow" }
        } else {
            val preparedBias = prepareBias(tensor)
            graph.addTensorToContext(preparedBias)
        }
    }

    private suspend fun appendPeepholes(tensor: KITensor?, graph: KIGraph) {
        if (tensor == null) {
            logger.warning { "Make peepholes part of the model, otherwise LSTM will be slow" }
        } else {
            val preparedPeepholes = preparePeepholes(tensor)
            graph.addTensorToContext(preparedPeepholes)
        }
    }

    override fun shouldApply(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>): Boolean {
        return operator is LSTM
    }

    override suspend fun transform(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        graph as KIGraph
        val initializers = graph.initializers as List<KITensor>

        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)
        val peepholesInit = initTensorByDefaultName("P", operator, initializers)

        appendWeights(weightsInit, graph)
        appendWeights(recurrentWeightsInit, graph)
        appendBias(biasInit, graph)
        appendPeepholes(peepholesInit, graph)
    }
}
