package io.kinference.tfjs.operators.layer.recurrent.gru

import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.tfjs.operators.layer.recurrent.LayerDirection

abstract class GRULayerBase(val hiddenSize: Int, val activations: List<String>, val direction: LayerDirection) {
    abstract suspend fun apply(
        input: NumberNDArrayTFJS, weights: NumberNDArrayTFJS, recurrentWeights: NumberNDArrayTFJS, bias: NumberNDArrayTFJS?,
        sequenceLength: NumberNDArrayTFJS?, initialHiddenState: NumberNDArrayTFJS?, linearBeforeReset: Boolean
    ): Pair<NumberNDArrayTFJS, NumberNDArrayTFJS>

    companion object {
        fun create(hiddenSize: Int, activations: List<String>, direction: LayerDirection) =
            when(direction) {
                LayerDirection.FORWARD, LayerDirection.REVERSE -> GRULayer(hiddenSize, activations, direction)
                LayerDirection.BIDIRECTIONAL -> BiGRULayer(hiddenSize, activations)
            }
    }
}
