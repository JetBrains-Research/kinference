package io.kinference.core.operators.layer.recurrent.gru

import io.kinference.core.operators.layer.recurrent.LayerDirection
import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.primitives.types.DataType

abstract class GRULayerBase(val hiddenSize: Int, val activations: List<String>, val direction: LayerDirection) {
    abstract suspend fun apply(
        input: NumberNDArrayCore, weights: NumberNDArrayCore, recurrentWeights: NumberNDArrayCore, bias: NumberNDArrayCore?,
        sequenceLength: IntNDArray?, initialHiddenState: NumberNDArrayCore?, dataType: DataType, linearBeforeReset: Boolean
    ): GRULayerOutput

    companion object {
        fun create(hiddenSize: Int, activations: List<String>, direction: LayerDirection) =
            when(direction) {
                LayerDirection.FORWARD, LayerDirection.REVERSE -> GRULayer(hiddenSize, activations, direction)
                LayerDirection.BIDIRECTIONAL -> BiGRULayer(hiddenSize, activations)
            }
    }
}
