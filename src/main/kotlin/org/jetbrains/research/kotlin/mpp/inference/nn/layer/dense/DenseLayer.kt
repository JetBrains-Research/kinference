package org.jetbrains.research.kotlin.mpp.inference.nn.layer.dense

import org.jetbrains.research.kotlin.mpp.inference.nn.activation.ActivatableVector
import org.jetbrains.research.kotlin.mpp.inference.nn.activation.ActivationFunction
import org.jetbrains.research.kotlin.mpp.inference.nn.layer.ActivatableLayer
import org.jetbrains.research.kotlin.mpp.inference.nn.parameters.DenseParameters
import scientifik.kmath.linear.BufferMatrix
import scientifik.kmath.linear.Point
import scientifik.kmath.structures.Matrix

class DenseLayer(
    name: String,
    override val params: DenseParameters,
    override val activationFunction: ActivationFunction
) : ActivatableLayer<Matrix<Double>>(name, params) {

    lateinit var inputArray: Point<Double>
    var outputArray: ActivatableVector = ActivatableVector(params.weights?.colNum ?: 0)

    override fun activate() {
        outputArray.forward(
            weights = params.weights!!,
            bias = params.biases,
            x = BufferMatrix(inputArray.size, 1, inputArray)
        )
        outputArray.activate(activationFunction)
    }

}
