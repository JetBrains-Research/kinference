package io.kinference.operators.layer.attention

import io.kinference.data.ONNXData
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ContextPrepare
import io.kinference.ndarray.logger
import io.kinference.operators.Operator
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
internal object AttentionContext: ContextPrepare() {
    private val logger = logger("Attention Initializer")

    override fun appendContext(context: Context, initializers: List<Tensor>, operator: Operator<ONNXData, ONNXData>) {
        val weightsInit = initTensorByDefaultName("weight", operator, initializers)
        val biasInit = initTensorByDefaultName("bias", operator, initializers)
        val numHeads = operator.getAttribute<Long>("num_heads").toInt()

        appendWeights(weightsInit, context, numHeads)
        appendBias(biasInit, context, numHeads)
    }

    internal fun prepareWeights(tensor: Tensor, numHeads: Int): Tensor {
        val shape = tensor.data.shape
        val headSize = shape[1] / 3 / numHeads
        val newShape = intArrayOf(shape[0], 3, numHeads, headSize)

        return tensor.data.toMutable().reshape(newShape).transpose(intArrayOf(1, 2, 0, 3)).asTensor("prepared_${tensor.info.name}")
    }

    internal fun prepareBias(tensor: Tensor, numHeads: Int): Tensor {
        val shape = tensor.data.shape
        val headSize = shape[0] / 3 / numHeads
        val newShape = intArrayOf(3, numHeads, headSize)
        return tensor.data.toMutable().reshape(newShape).asTensor("prepared_${tensor.info.name}")
    }

    private fun appendWeights(tensor: Tensor?, context: Context, numHeads: Int) {
        if (tensor == null) {
            logger.warn { "Make the weights part of the model, otherwise the Attention will be slow" }
        } else {
            val preparedWeights = prepareWeights(tensor, numHeads)
            context.putValue(preparedWeights.info.name, preparedWeights)
        }
    }

    private fun appendBias(tensor: Tensor?, context: Context, numHeads: Int) {
        if (tensor == null) {
            logger.warn { "Make the bias part of the model, otherwise the Attention will be slow" }
        } else {
            val preparedBias = prepareBias(tensor, numHeads)
            context.putValue(preparedBias.info.name, preparedBias)
        }
    }
}
