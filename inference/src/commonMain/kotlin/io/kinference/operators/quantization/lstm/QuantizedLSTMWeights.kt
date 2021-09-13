package io.kinference.operators.quantization.lstm

import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.operators.layer.recurrent.lstm.AbstractLSTMWeights

class QuantizedLSTMWeights(data: NumberNDArray, val scale: FloatNDArray, val zeroPoint: NumberNDArray): AbstractLSTMWeights(data) {
    override fun view(dim: Int): QuantizedLSTMWeights {
        return if (data.rank == 4)
            QuantizedLSTMWeights(data.view(dim), scale.view(dim), zeroPoint.view(dim))
        else
            QuantizedLSTMWeights(data.view(dim), scale, zeroPoint)
    }
}
