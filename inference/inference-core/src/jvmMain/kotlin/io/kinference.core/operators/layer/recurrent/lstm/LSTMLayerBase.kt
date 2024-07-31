package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.core.operators.layer.recurrent.LayerDirection
import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.primitives.types.DataType


abstract class LSTMLayerBase(val hiddenSize: Int, val activations: List<String>, val direction: LayerDirection) {

    abstract suspend fun apply(
        input: AbstractLSTMInput,
        weights: AbstractLSTMWeights,
        recurrentWeights: AbstractLSTMWeights,
        bias: NumberNDArrayCore?,
        sequenceLens: IntNDArray?,
        initialHiddenState: NumberNDArrayCore?,
        initialCellState: NumberNDArrayCore?,
        peepholes: NumberNDArrayCore?,
        dataType: DataType
    ): LSTMLayerOutput

    companion object {
        fun create(hiddenSize: Int, activations: List<String>, direction: LayerDirection) =
            when(direction) {
                LayerDirection.FORWARD, LayerDirection.REVERSE -> LSTMLayer(hiddenSize, activations, direction)
                LayerDirection.BIDIRECTIONAL -> BiLSTMLayer(hiddenSize, activations)
            }

    }
}
