package org.jetbrains.research.kotlin.mpp.inference.nn.layer.lstm

import org.jetbrains.research.kotlin.mpp.inference.nn.activation.ActivationFunction
import org.jetbrains.research.kotlin.mpp.inference.nn.layer.RecurrentLayer
import org.jetbrains.research.kotlin.mpp.inference.nn.parameters.LSTMParameters
import scientifik.kmath.linear.*
import scientifik.kmath.structures.Matrix
import scientifik.kmath.structures.Buffer
import scientifik.kmath.linear.MatrixContext

class LSTMLayer(
    name: String,
    override val params: LSTMParameters
) : RecurrentLayer<Matrix<Double>>(name, params) {
    lateinit var inputVectors: Array<Point<Double>>
    private var outputSize = params.Wf?.rowNum ?: 0
    val outputVectors: Array<Point<Double>> = emptyArray()


    protected fun entrywiseMultiply(first: Point<Double>, second: Point<Double>) =
        Buffer.Companion.real(outputSize) {first[it] * second[it]}

    override fun activate() {
        val vectorSpace = VectorSpace.real(outputSize)

        var prevH: Point<Double> = vectorSpace.zero
        var prevC: Point<Double> = vectorSpace.zero

        for (i in 1..outputSize){
            val inputVector = inputVectors[i]

            val ft1: Point<Double>
            val ft2: Point<Double>

            val it1: Point<Double>
            val it2: Point<Double>

            val ot1: Point<Double>
            val ot2: Point<Double>

            val ct1: Point<Double>
            val ct2: Point<Double>

            with (MatrixContext.real) {
                ft1 = params.Wf!!.dot(inputVector)
                ft2 = params.Uf!!.dot(prevH)

                it1 = params.Wi!!.dot(inputVector)
                it2 = params.Ui!!.dot(prevH)

                ot1 = params.Wo!!.dot(inputVector)
                ot2 = params.Uo!!.dot(prevH)

                ct1 = params.Wc!!.dot(inputVector)
                ct2 = params.Uc!!.dot(prevH)
            }

            var ft = vectorSpace.add(vectorSpace.add(ft1, ft2), params.bf!!)
            var it = vectorSpace.add(vectorSpace.add(it1, it2), params.bi!!)
            var ot = vectorSpace.add(vectorSpace.add(ot1, ot2), params.bo!!)
            var nowC = vectorSpace.add(vectorSpace.add(ct1, ct2), params.bc!!)


            ft = ActivationFunction.Sigmoid(ft).asPoint()
            it = ActivationFunction.Sigmoid(it).asPoint()
            ot = ActivationFunction.Sigmoid(ot).asPoint()
            nowC = ActivationFunction.Tanh(nowC).asPoint()
            nowC = vectorSpace.add(entrywiseMultiply(ft, prevC), entrywiseMultiply(nowC, it))

            outputVectors[i] = entrywiseMultiply(ot, ActivationFunction.Tanh(nowC).asPoint())
            prevH = outputVectors[i]
            prevC = nowC
        }
    }
}
