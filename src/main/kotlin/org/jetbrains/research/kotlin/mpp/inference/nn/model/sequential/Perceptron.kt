package org.jetbrains.research.kotlin.mpp.inference.nn.model.sequential

import org.jetbrains.research.kotlin.mpp.inference.nn.layer.Layer
import org.jetbrains.research.kotlin.mpp.inference.nn.layer.dense.DenseLayer
import scientifik.kmath.linear.Point
import scientifik.kmath.linear.asPoint

open class Perceptron(
    name: String,
    override val layers: List<DenseLayer>,
    batchInputShape: List<Int?>
) : SequentialModel<FloatArray, Float>(name, layers, batchInputShape) {
    companion object {
        @Suppress("UNCHECKED_CAST")
        fun create(name: String, layers: List<Layer<*>>, batchInputShape: List<Int?>?): Perceptron {
            require(batchInputShape != null) { "Model input shape is unspecified" }
            require(batchInputShape.filterNotNull().size == 1) { "Input should be one-dimensional" }

            layers as List<DenseLayer>

            return Perceptron(name, layers, batchInputShape)
        }
    }

    override fun predict(input: FloatArray): Float {
        require(batchInputShape!!.filterNotNull().single() == input.size) { "Unmatched input shapes" }

        layers.first().let {
            it.inputArray = Point.real(input.size) { i -> input[i].toDouble() }
            it.activate()
        }

        layers.zipWithNext { prev, cur ->
            cur.inputArray = prev.outputArray.values.asPoint()
            cur.activate()
        }

        return layers.last().outputArray.values[0, 0].toFloat()
    }

}
