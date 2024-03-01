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
import io.kinference.tfjs.operators.conv.Conv
import io.kinference.utils.LoggerFactory

object ConvContextRule : PrepareContextRule<TFJSData<*>>(operatorName = "Conv") {
    private val logger = LoggerFactory.create("io.kinference.tfjs.optimizer.rules.context.ConvContextRule")

    internal suspend fun prepareWeights(tensor: TFJSTensor): TFJSTensor {
        val transposeShape = when(val rank = tensor.data.rank) {
            3 -> intArrayOf(2, 1, 0)
            4 -> intArrayOf(2, 3, 1, 0)
            5 -> intArrayOf(2, 3, 4, 1, 0)
            else -> error("Unsupported tensor rank for convolution: ${rank}. Supported ranks are: 3, 4 and 5")
        }

        return tidyNDArray {
            tensor.data.transpose(transposeShape)
        }.asTensor(optName(tensor.name!!))
    }

    private suspend fun appendWeights(tensor: TFJSTensor?, graph: TFJSGraph, operator: Operator<TFJSData<*>, TFJSData<*>>) {
        if (tensor == null) {
            logger.warning { "Add weights to the model's initializers, otherwise the Conv operator inference will be slower than expected" }
        } else {
            val preparedWeights = prepareWeights(tensor)
            graph.addTensorToContext(preparedWeights)

            operator.renameInput(tensor.name!!, preparedWeights.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    override fun shouldApply(graph: Graph<TFJSData<*>>, operator: Operator<TFJSData<*>, TFJSData<*>>): Boolean {
        return operator is Conv
    }

    override suspend fun transform(graph: Graph<TFJSData<*>>, operator: Operator<TFJSData<*>, TFJSData<*>>) {
        graph as TFJSGraph

        val initializers = graph.getInitializers() as List<TFJSTensor>
        val weightsInit = ConvContextRule.initTensorByDefaultName("W", operator, initializers)

        appendWeights(weightsInit, graph, operator)
    }
}
