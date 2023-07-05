package io.kinference.core.optimizer.rules.context

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.operators.math.MatMulInteger
import io.kinference.graph.Graph
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.mapTo
import io.kinference.ndarray.arrays.tiled.IntTiledArray
import io.kinference.ndarray.extensions.tryZeroPoint
import io.kinference.operator.Operator
import io.kinference.optimizer.GraphOptimizer.Companion.optName
import io.kinference.optimizer.rules.context.PrepareContextRule

object MatMulIntegerContextRule : PrepareContextRule<KIONNXData<*>>(operatorName = "MatMulInteger") {
    private fun NumberNDArray.toIntNDArray(): IntNDArray {
        val result = IntNDArray(IntTiledArray(this.strides), strides)
        when (this) {
            is UByteNDArray -> {
                this.array.pointer().mapTo(result.array.pointer(), linearSize) { it.toInt() }
            }
            is ByteNDArray -> {
                this.array.pointer().mapTo(result.array.pointer(), linearSize) { it.toInt() }
            }
            else -> error("Unsupported data type: $type")
        }

        return result
    }

    internal suspend fun prepareTensor(tensor: KITensor, zeroPoint: KITensor?): KITensor {
        val preparedTensor = if (zeroPoint == null)
            (tensor.data as NumberNDArrayCore).toIntNDArray()
        else
            (tensor.data as NumberNDArrayCore).tryZeroPoint(zeroPoint.data as NumberNDArrayCore)

        return preparedTensor.asTensor(optName(tensor.name))
    }

    private suspend fun appendTensor(tensor: KITensor?, zeroPoint: KITensor?, graph: KIGraph, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        if (tensor != null) {
            val preparedTensor = prepareTensor(tensor, zeroPoint)
            graph.addTensorToContext(preparedTensor)

            operator.renameInput(tensor.name!!, preparedTensor.name!!)
            tryRemoveDefaultInitializer(graph, tensor.name!!)
        }
    }

    override fun shouldApply(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>): Boolean {
        return operator is MatMulInteger
    }

    override suspend fun transform(graph: Graph<KIONNXData<*>>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
        graph as KIGraph
        val initializers = graph.initializers as List<KITensor>

        val leftTensor = initTensorByDefaultName("A", operator, initializers)
        val rightTensor = initTensorByDefaultName("B", operator, initializers)
        val leftZeroPoint = initTensorByDefaultName("a_zero_point", operator, initializers)
        val rightZeroPoint = initTensorByDefaultName("b_zero_point", operator, initializers)

        appendTensor(leftTensor, leftZeroPoint, graph, operator)
        appendTensor(rightTensor, rightZeroPoint, graph, operator)
    }
}
