package io.kinference.operators.layer.recurrent.gru

import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.primitives.types.DataType

abstract class GRULayerBase(val hiddenSize: Int, val activations: List<String>, val direction: String) {
    abstract fun apply(input: NumberNDArray, weights: NumberNDArray, recurrentWeights: NumberNDArray, bias: NumberNDArray?, sequenceLens: IntNDArray?,
                       initialHiddenState: NumberNDArray?, dataType: DataType, linearBeforeReset: Boolean): Pair<NumberNDArray, NumberNDArray>

    companion object {
        fun create(hiddenSize: Int, activations: List<String>, direction: String) =
            when(direction) {
                "forward", "reverse" -> GRULayer(hiddenSize, activations, direction)
                "bidirectional" -> BiGRULayer(hiddenSize, activations)
                else -> error("Bad direction attribute")
            }
    }
}
