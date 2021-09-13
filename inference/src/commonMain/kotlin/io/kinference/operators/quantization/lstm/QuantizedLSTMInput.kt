package io.kinference.operators.quantization.lstm

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.quantizeMatMul
import io.kinference.operators.layer.recurrent.lstm.AbstractLSTMInput
import io.kinference.operators.layer.recurrent.lstm.AbstractLSTMWeights
import io.kinference.operators.quantization.DynamicQuantizeLinear.Companion.dynamicQuantize
import kotlin.time.ExperimentalTime

class QuantizedLSTMInput(data: NumberNDArray, val scale: FloatNDArray, val zeroPoint: NumberNDArray): AbstractLSTMInput(data) {
    override fun view(vararg dims: Int): QuantizedLSTMInput = QuantizedLSTMInput(data.view(*dims), scale, zeroPoint)

    override fun dot(weights: AbstractLSTMWeights, destination: MutableNumberNDArray) {
        when(weights) {
            is QuantizedLSTMWeights -> quantizeMatMul(data, weights.data, zeroPoint, weights.zeroPoint, scale, weights.scale, destination as MutableFloatNDArray)
            else -> error("Unsupported operation")
        }
    }

    override fun recreate(data: NumberNDArray): QuantizedLSTMInput {
        require(data is FloatNDArray)
        return create(data)
    }

    companion object {
        @OptIn(ExperimentalTime::class)
        fun create(data: FloatNDArray): QuantizedLSTMInput {
            val (quantizedData, scale, zeroPoint) = data.dynamicQuantize()
            return QuantizedLSTMInput(quantizedData, scale, zeroPoint)
        }
    }
}
