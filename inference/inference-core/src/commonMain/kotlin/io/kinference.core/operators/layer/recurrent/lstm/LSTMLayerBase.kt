package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType


abstract class LSTMLayerBase(val hiddenSize: Int, val activations: List<String>, val direction: String) {

    abstract fun apply(input: AbstractLSTMInput, weights: AbstractLSTMWeights, recurrentWeights: AbstractLSTMWeights, bias: NumberNDArray?, sequenceLens: IntNDArray?,
                       initialHiddenState: NumberNDArray?, initialCellState: NumberNDArray?, peepholes: NumberNDArray?, dataType: DataType)
    : Triple<NumberNDArray, NumberNDArray, NumberNDArray>

    companion object {
        fun create(hiddenSize: Int, activations: List<String>, direction: String) =
            when(direction) {
                "forward", "reverse" -> LSTMLayer(hiddenSize, activations, direction)
                "bidirectional" -> BiLSTMLayer(hiddenSize, activations)
                else -> error("Bad direction attribute")
            }

    }
}
