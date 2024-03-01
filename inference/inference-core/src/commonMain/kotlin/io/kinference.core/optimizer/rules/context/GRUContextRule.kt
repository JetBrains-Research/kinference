package io.kinference.core.optimizer.rules.context

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.operators.layer.recurrent.gru.GRU
import io.kinference.graph.Graph
import io.kinference.operator.Operator
import io.kinference.optimizer.GraphOptimizer.Companion.optName
import io.kinference.optimizer.rules.context.PrepareContextRule
import io.kinference.utils.LoggerFactory

object GRUContextRule : PrepareContextRule<KIONNXData<*>>(operatorName = "GRU") {
    private val logger = LoggerFactory.create("io.kinference.core.optimizer.rules.context.GRUContextRule")

    internal suspend fun prepareWeights(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 3, shape[1] / 3, shape[2])
        return tensor.data.reshape(newShape).transpose(intArrayOf(0, 1, 3, 2)).asTensor(optName(tensor.name))
    }

    internal suspend fun prepareBias(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 6, shape[1] / 6)
        return tensor.data.reshape(newShape).asTensor(optName(tensor.name))
    }

    private suspend fun appendWeights(tensor: KITensor?, graph: KIGraph, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        if (tensor == null) {
            logger.warning { "Add weights to the model's initializers, otherwise the GRU operator inference will be slower than expected" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            graph.addTensorToContext(preparedWeights)

            operator.renameInput(tensor.name!!, preparedWeights.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    private suspend fun appendBias(tensor: KITensor?, graph: KIGraph, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        if (tensor == null) {
            logger.warning { "Add bias to the model's initializers, otherwise the GRU operator inference will be slower than expected" }
        } else {
            val preparedBias = prepareBias(tensor)
            graph.addTensorToContext(preparedBias)

            operator.renameInput(tensor.name!!, preparedBias.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    override fun shouldApply(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>): Boolean {
        return operator is GRU
    }

    override suspend fun transform(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        graph as KIGraph
        val initializers = graph.getInitializers() as List<KITensor>

        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)

        appendWeights(weightsInit, graph, operator)
        appendWeights(recurrentWeightsInit, graph, operator)
        appendBias(biasInit, graph, operator)
    }
}
