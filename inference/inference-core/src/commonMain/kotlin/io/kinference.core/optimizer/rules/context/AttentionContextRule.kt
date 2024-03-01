package io.kinference.core.optimizer.rules.context

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.operators.layer.attention.Attention
import io.kinference.graph.Graph
import io.kinference.operator.Operator
import io.kinference.optimizer.GraphOptimizer.Companion.optName
import io.kinference.optimizer.rules.context.PrepareContextRule
import io.kinference.utils.LoggerFactory

object AttentionContextRule : PrepareContextRule<KIONNXData<*>>(operatorName = "Attention") {
    private val logger = LoggerFactory.create("io.kinference.core.optimizer.rules.context.AttentionContextRule")

    internal suspend fun prepareWeights(tensor: KITensor, numHeads: Int): KITensor {
        val shape = tensor.data.shape
        val headSize = shape[1] / 3 / numHeads
        val newShape = intArrayOf(shape[0], 3, numHeads, headSize)

        val prepared = tensor.data.reshape(newShape).transpose(intArrayOf(1, 2, 0, 3))

        return prepared.asTensor(optName(tensor.name))
    }

    internal suspend fun prepareBias(tensor: KITensor, numHeads: Int): KITensor {
        val shape = tensor.data.shape
        val headSize = shape[0] / 3 / numHeads
        val newShape = intArrayOf(3, numHeads, headSize)
        return tensor.data.reshape(newShape).asTensor(optName(tensor.name))
    }

    private suspend fun appendWeights(tensor: KITensor?, graph: KIGraph, operator: Operator<KIONNXData<*>, KIONNXData<*>>, numHeads: Int) {
        if (tensor == null) {
            logger.warning { "Add weights to the model's initializers, otherwise the Attention operator inference will be slower than expected" }
        } else {
            val preparedWeights = prepareWeights(tensor, numHeads)
            graph.addTensorToContext(preparedWeights)

            operator.renameInput(tensor.name!!, preparedWeights.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    private suspend fun appendBias(tensor: KITensor?, graph: KIGraph, operator: Operator<KIONNXData<*>, KIONNXData<*>>, numHeads: Int) {
        if (tensor == null) {
            logger.warning { "Add bias to the model's initializers, otherwise the Attention operator inference will be slower than expected" }
        } else {
            val preparedBias = prepareBias(tensor, numHeads)
            graph.addTensorToContext(preparedBias)

            operator.renameInput(tensor.name!!, preparedBias.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    override fun shouldApply(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>): Boolean {
        return operator is Attention
    }

    override suspend fun transform(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        graph as KIGraph
        val initializers = graph.getInitializers() as List<KITensor>

        val weightsInit = initTensorByDefaultName("weight", operator, initializers)
        val biasInit = initTensorByDefaultName("bias", operator, initializers)
        val numHeads = operator.getAttribute<Long>("num_heads").toInt()

        appendWeights(weightsInit, graph, operator, numHeads)
        appendBias(biasInit, graph, operator, numHeads)
    }
}
