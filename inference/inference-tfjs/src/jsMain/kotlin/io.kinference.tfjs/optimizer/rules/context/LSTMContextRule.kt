package io.kinference.tfjs.optimizer.rules.context

import io.kinference.graph.Graph
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.Operator
import io.kinference.optimizer.GraphOptimizer.Companion.optName
import io.kinference.optimizer.rules.context.PrepareContextRule
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.graph.TFJSGraph
import io.kinference.tfjs.operators.layer.recurrent.lstm.LSTM
import io.kinference.utils.LoggerFactory

object LSTMContextRule : PrepareContextRule<TFJSData<*>>(operatorName = "LSTM") {
    private val logger = LoggerFactory.create("io.kinference.tfjs.optimizer.rules.context.LSTMContextRule")

    internal suspend fun prepareWeights(tensor: TFJSTensor): TFJSTensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 4, shape[1] / 4, shape[2])
        val transposeShape = intArrayOf(0, 1, 3, 2)

        return tidyNDArray {
            tensor.data.reshape(newShape).transpose(transposeShape)
        }.asTensor(optName(tensor.name!!))
    }

    internal suspend fun prepareBias(tensor: TFJSTensor): TFJSTensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 8, shape[1] / 8)
        return tensor.data.reshape(newShape).asTensor(optName(tensor.name!!))
    }

    internal suspend fun preparePeepholes(tensor: TFJSTensor): TFJSTensor {
        val shape = tensor.data.shape
        val newShape = intArrayOf(shape[0], 3, shape[1] / 3)
        return tensor.data.reshape(newShape).asTensor(optName(tensor.name!!))
    }

    private suspend fun appendWeights(tensor: TFJSTensor?, graph: TFJSGraph, operator: Operator<TFJSData<*>, TFJSData<*>>) {
        if (tensor == null) {
            logger.warning { "Add weights to the model's initializers, otherwise the LSTM operator inference will be slower than expected" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            graph.addTensorToContext(preparedWeights)

            operator.renameInput(tensor.name!!, preparedWeights.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    private suspend fun appendBias(tensor: TFJSTensor?, graph: TFJSGraph, operator: Operator<TFJSData<*>, TFJSData<*>>) {
        if (tensor == null) {
            logger.warning { "Add bias to the model's initializers, otherwise the LSTM operator inference will be slower than expected" }
        } else {
            val preparedBias = prepareBias(tensor)
            graph.addTensorToContext(preparedBias)

            operator.renameInput(tensor.name!!, preparedBias.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    private suspend fun appendPeepholes(tensor: TFJSTensor?, graph: TFJSGraph, operator: Operator<TFJSData<*>, TFJSData<*>>) {
        if (tensor == null) {
            logger.warning { "Add peepholes to the model's initializers, otherwise the LSTM operator inference will be slower than expected" }
        } else {
            val preparedPeepholes = preparePeepholes(tensor)
            graph.addTensorToContext(preparedPeepholes)

            operator.renameInput(tensor.name!!, preparedPeepholes.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    override fun shouldApply(graph: Graph<TFJSData<*>>, operator: Operator<TFJSData<*>, TFJSData<*>>): Boolean {
        return operator is LSTM
    }

    override suspend fun transform(graph: Graph<TFJSData<*>>, operator: Operator<TFJSData<*>, TFJSData<*>>) {
        graph as TFJSGraph
        val initializers = graph.getInitializers() as List<TFJSTensor>

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
