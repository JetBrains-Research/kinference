package io.kinference.core.optimizer.rules.context

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.operators.layer.recurrent.gru.GRU
import io.kinference.graph.Graph
import io.kinference.operator.Operator
import io.kinference.utils.LoggerFactory

object GRUContextRule : PrepareContextRule(operatorName = "GRU") {
    private val logger = LoggerFactory.create("io.kinference.core.optimizer.rules.context.GRUContextRule")

    internal suspend fun prepareWeights(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 3, shape[1] / 3, shape[2])
        return tensor.data.reshape(newShape).transpose(intArrayOf(0, 1, 3, 2)).asTensor("${PREFIX}_${tensor.name}")
    }

    internal suspend fun prepareBias(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 6, shape[1] / 6)
        return tensor.data.reshape(newShape).asTensor("${PREFIX}_${tensor.name}")
    }

    private suspend fun appendWeights(tensor: KITensor?, graph: KIGraph) {
        if (tensor == null) {
            logger.warning { "Make the weights part of the model, otherwise the GRU will be slow" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            graph.addTensorToContext(preparedWeights)
        }
    }

    private suspend fun appendBias(tensor: KITensor?, graph: KIGraph) {
        if (tensor == null) {
            logger.warning { "Make the bias part of the model, otherwise the GRU will be slow" }
        } else {
            val preparedBias = prepareBias(tensor)
            graph.addTensorToContext(preparedBias)
        }
    }

    override fun shouldApply(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>): Boolean {
        return operator is GRU
    }

    override suspend fun transform(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        graph as KIGraph
        val initializers = graph.initializers as List<KITensor>

        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)

        appendWeights(weightsInit, graph)
        appendWeights(recurrentWeightsInit, graph)
        appendBias(biasInit, graph)
    }
}
