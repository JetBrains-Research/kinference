package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType


abstract class LSTMLayerBase(val hiddenSize: Int, val activations: List<String>, val direction: String) {

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
        fun create(hiddenSize: Int, activations: List<String>, direction: String) =
            when(direction) {
                "forward", "reverse" -> LSTMLayer(hiddenSize, activations, direction)
                "bidirectional" -> BiLSTMLayer(hiddenSize, activations)
                else -> error("Bad direction attribute")
            }

    }
}
