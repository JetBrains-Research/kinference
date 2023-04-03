package io.kinference.core.operators.layer.recurrent.gru

import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.primitives.types.DataType

abstract class GRULayerBase(val hiddenSize: Int, val activations: List<String>, val direction: String) {
    abstract suspend fun apply(
        input: NumberNDArrayCore, weights: NumberNDArrayCore, recurrentWeights: NumberNDArrayCore, bias: NumberNDArrayCore?,
        sequenceLength: IntNDArray?, initialHiddenState: NumberNDArrayCore?, dataType: DataType, linearBeforeReset: Boolean
    ): Pair<NumberNDArrayCore, NumberNDArrayCore>

    companion object {
        fun create(hiddenSize: Int, activations: List<String>, direction: String) =
            when(direction) {
                "forward", "reverse" -> GRULayer(hiddenSize, activations, direction)
                "bidirectional" -> BiGRULayer(hiddenSize, activations)
                else -> error("Bad direction attribute")
            }
    }
}
