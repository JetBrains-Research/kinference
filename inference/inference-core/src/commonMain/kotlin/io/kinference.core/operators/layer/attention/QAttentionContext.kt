package io.kinference.core.operators.layer.attention

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.ContextPrepare
import io.kinference.graph.GraphContext
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.ndarray.extensions.tryDequantize
import io.kinference.operator.Operator

object QAttentionContext: ContextPrepare() {
    override suspend fun appendContext(
        context: GraphContext<KIONNXData<*>>,
        initializers: List<KITensor>,
        operator: Operator<KIONNXData<*>, KIONNXData<*>>
    ) {
        val weightsTensor = initTensorByDefaultName("weight", operator, initializers)
        val biasTensor = initTensorByDefaultName("bias", operator, initializers)
        val weightScale = initTensorByDefaultName("weight_scale", operator, initializers)
        val weightZeroPoint = initTensorByDefaultName("weight_zero_point", operator, initializers)
        val numHeads = operator.getAttribute<Long>("num_heads").toInt()

        appendWeights(weightsTensor, weightScale, weightZeroPoint, numHeads, context)
        appendBias(biasTensor, context, numHeads)
    }


    internal suspend fun prepareWeights(tensor: KITensor, scale: KITensor, zeroPoint: KITensor?, numHeads: Int): KITensor {
        val shape = tensor.data.shape
        val headSize = shape[1] / 3 / numHeads
        val newShape = intArrayOf(shape[0], 3, numHeads, headSize)

        val dequantData = (tensor.data as NumberNDArrayCore).tryDequantize(zeroPoint?.data as NumberNDArrayCore?, scale.data as FloatNDArray)

        return dequantData.reshape(newShape).transpose(intArrayOf(1, 2, 0, 3)).asTensor("prepared_${tensor.name}")
    }

    private suspend fun appendWeights(tensor: KITensor?, scale: KITensor?, zeroPoint: KITensor?, numHeads: Int, context: GraphContext<KIONNXData<*>>) {
        if (tensor != null && scale != null) {
            val preparedWeights = prepareWeights(tensor, scale, zeroPoint,numHeads)
            context.putValue(preparedWeights.name!!, preparedWeights)
        }
    }

    private suspend fun appendBias(tensor: KITensor?, context: GraphContext<KIONNXData<*>>, numHeads: Int) {
        if (tensor != null) {
            val preparedBias = AttentionContext.prepareBias(tensor, numHeads)
            context.putValue(preparedBias.name!!, preparedBias)
        }
    }
}
