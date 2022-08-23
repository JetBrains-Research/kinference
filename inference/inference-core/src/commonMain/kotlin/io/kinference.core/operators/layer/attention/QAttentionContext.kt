package io.kinference.core.operators.layer.attention

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.ContextPrepare
import io.kinference.core.graph.KIContext
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.ndarray.extensions.tryDequantize
import io.kinference.operator.Operator
import io.kinference.protobuf.message.TypeProto

object QAttentionContext: ContextPrepare() {
    override fun appendContext(
        context: KIContext,
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


    internal fun prepareWeights(tensor: KITensor, scale: KITensor, zeroPoint: KITensor?, numHeads: Int): KITensor {
        val shape = tensor.data.shape
        val headSize = shape[1] / 3 / numHeads
        val newShape = intArrayOf(shape[0], 3, numHeads, headSize)

        val dequantData = (tensor.data as NumberNDArray).tryDequantize(zeroPoint?.data as NumberNDArray?, scale.data as FloatNDArray)

        return dequantData.reshape(newShape).transpose(intArrayOf(1, 2, 0, 3)).asTensor("prepared_${tensor.name}")
    }

    private fun appendWeights(tensor: KITensor?, scale: KITensor?, zeroPoint: KITensor?, numHeads: Int, context: KIContext) {
        if (tensor != null && scale != null) {
            val preparedWeights = prepareWeights(tensor, scale, zeroPoint,numHeads)
            context.putValue(preparedWeights.name!!, preparedWeights)
        }
    }

    private fun appendBias(tensor: KITensor?, context: KIContext, numHeads: Int) {
        if (tensor != null) {
            val preparedBias = AttentionContext.prepareBias(tensor, numHeads)
            context.putValue(preparedBias.name!!, preparedBias)
        }
    }
}
