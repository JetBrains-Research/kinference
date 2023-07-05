package io.kinference.core.optimizer.rules.context

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.operators.layer.attention.*
import io.kinference.graph.Graph
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.ndarray.extensions.tryDequantize
import io.kinference.operator.Operator
import io.kinference.optimizer.GraphOptimizer.Companion.optName
import io.kinference.utils.LoggerFactory

object QAttentionContextRule : PrepareContextRule(operatorName = "QAttention") {
    private val logger = LoggerFactory.create("io.kinference.core.optimizer.rules.context.QAttentionContextRule")

    internal suspend fun prepareWeights(tensor: KITensor, scale: KITensor, zeroPoint: KITensor?, numHeads: Int): KITensor {
        val shape = tensor.data.shape
        val headSize = shape[1] / 3 / numHeads
        val newShape = intArrayOf(shape[0], 3, numHeads, headSize)

        val dequantData = (tensor.data as NumberNDArrayCore)
            .tryDequantize(zeroPoint?.data as NumberNDArrayCore?, scale.data as FloatNDArray)

        return dequantData.reshape(newShape).transpose(intArrayOf(1, 2, 0, 3)).asTensor(optName(tensor.name))
    }

    private suspend fun appendWeights(tensor: KITensor?, scale: KITensor?, zeroPoint: KITensor?, numHeads: Int, graph: KIGraph) {
        if (tensor != null && scale != null) {
            val preparedWeights = prepareWeights(tensor, scale, zeroPoint,numHeads)
            graph.addTensorToContext(preparedWeights)
        } else {
            logger.warning { "Add weights to the model's initializers, otherwise the QAttention operator inference will be slower than expected" }
        }
    }

    private suspend fun appendBias(tensor: KITensor?, graph: KIGraph, numHeads: Int) {
        if (tensor == null) {
            logger.warning { "Add bias to the model's initializers, otherwise the QAttention operator inference will be slower than expected" }
        } else {
            val preparedBias = AttentionContextRule.prepareBias(tensor, numHeads)
            graph.addTensorToContext(preparedBias)
        }
    }

    override fun shouldApply(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>): Boolean {
        return operator is QAttention
    }

    override suspend fun transform(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        graph as KIGraph
        val initializers = graph.initializers as List<KITensor>

        val weightsTensor = initTensorByDefaultName("weight", operator, initializers)
        val biasTensor = initTensorByDefaultName("bias", operator, initializers)
        val weightScale = initTensorByDefaultName("weight_scale", operator, initializers)
        val weightZeroPoint = initTensorByDefaultName("weight_zero_point", operator, initializers)
        val numHeads = operator.getAttribute<Long>("num_heads").toInt()

        appendWeights(weightsTensor, weightScale, weightZeroPoint, numHeads, graph)
        appendBias(biasTensor, graph, numHeads)
    }
}
