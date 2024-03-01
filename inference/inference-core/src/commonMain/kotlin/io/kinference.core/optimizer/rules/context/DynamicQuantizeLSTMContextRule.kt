package io.kinference.core.optimizer.rules.context

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.operators.quantization.lstm.DynamicQuantizeLSTM
import io.kinference.graph.Graph
import io.kinference.operator.Operator
import io.kinference.optimizer.GraphOptimizer.Companion.optName
import io.kinference.optimizer.rules.context.PrepareContextRule
import io.kinference.utils.LoggerFactory

object DynamicQuantizeLSTMContextRule : PrepareContextRule<KIONNXData<*>>(operatorName = "DynamicQuantizeLSTM") {
    private val logger = LoggerFactory.create("io.kinference.core.optimizer.rules.context.DynamicQuantizeLSTMContextRule")

    internal suspend fun prepareWeights(tensor: KITensor): KITensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], shape[1], 4, shape[2] / 4)
        return tensor.data
            .reshape(newShape)
            .transpose(intArrayOf(0, 2, 1, 3)).asTensor(optName(tensor.name))
    }

    private suspend fun appendWeights(tensor: KITensor?, graph: KIGraph, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        if (tensor == null) {
            logger.warning { "Add weights to the model's initializers, otherwise the DynamicQuantizeLSTM operator inference will be slower than expected" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            graph.addTensorToContext(preparedWeights)

            operator.renameInput(tensor.name!!, preparedWeights.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    private suspend fun appendBias(tensor: KITensor?, graph: KIGraph, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        if (tensor == null) {
            logger.warning { "Add bias to the model's initializers, otherwise the DynamicQuantizeLSTM operator inference will be slower than expected" }
        } else {
            val preparedBias = LSTMContextRule.prepareBias(tensor)
            graph.addTensorToContext(preparedBias)

            operator.renameInput(tensor.name!!, preparedBias.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    private suspend fun appendPeepholes(tensor: KITensor?, graph: KIGraph, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        if (tensor == null) {
            logger.warning { "Add peepholes to the model's initializers, otherwise the DynamicQuantizeLSTM operator inference will be slower than expected" }
        } else {
            val preparedPeepholes = LSTMContextRule.preparePeepholes(tensor)
            graph.addTensorToContext(preparedPeepholes)

            operator.renameInput(tensor.name!!, preparedPeepholes.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    override fun shouldApply(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>): Boolean {
        return operator is DynamicQuantizeLSTM
    }

    override suspend fun transform(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        graph as KIGraph
        val initializers = graph.getInitializers() as List<KITensor>

        val weightsInit = initTensorByDefaultName("W", operator, initializers)
        val recurrentWeightsInit = initTensorByDefaultName("R", operator, initializers)
        val biasInit = initTensorByDefaultName("B", operator, initializers)
        val peepholesInit = initTensorByDefaultName("P", operator, initializers)

        appendWeights(weightsInit, graph, operator)
        appendWeights(recurrentWeightsInit, graph, operator)
        appendBias(biasInit, graph, operator)
        appendPeepholes(peepholesInit, graph, operator)
    }
}
