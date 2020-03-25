package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import org.jetbrains.research.kotlin.mpp.inference.operators.activations.Activation
import org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.RecurrentLayer
import org.jetbrains.research.kotlin.mpp.inference.space.TensorRing
import org.jetbrains.research.kotlin.mpp.inference.space.resolveSpaceWithKClass
import org.jetbrains.research.kotlin.mpp.inference.space.toIntArray
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.types.resolveKClass
import scientifik.kmath.structures.asIterable

class LSTMLayer<T : Number> : RecurrentLayer<T>() {
    @Suppress("UNCHECKED_CAST")
    override fun apply(inputs: Collection<Tensor<T>>): Collection<Tensor<T>> {
        require(inputs.size == 4) { "Applicable only for four arguments" }

        val inputList = inputs.toList()

        val inputTensor = inputList[0]
        val weights = inputList[1].as2DCollection().first()
        val recWeights = inputList[2].as2DCollection().first()
        val bias = inputList[3].transpose()

        val hiddenSize = recWeights.data.shape[2]
        val batchSize = inputTensor.data.shape[1]

        val mySpace = resolveSpaceWithKClass(inputTensor.type!!.resolveKClass(), intArrayOf(hiddenSize, batchSize))
        mySpace as TensorRing<T>

        var prevH = Tensor(null, mySpace.zero, inputTensor.type, mySpace)
        var prevC = Tensor(null, mySpace.zero, inputTensor.type, mySpace)
        val resH = mutableListOf<Tensor<T>>()

        for (inputVector in inputTensor.as2DCollection()) {
            var gates = weights.dot(inputVector.transpose()) + recWeights.dot(prevH.transpose())
            gates = addBiases(gates!!, bias, hiddenSize)
            val gatesCollection = getGates(gates, batchSize.toLong(), hiddenSize.toLong())
            var inputGate = gatesCollection[0]
            var outputGate = gatesCollection[1]
            var forgetGate = gatesCollection[2]
            var cellGate = gatesCollection[3]

            inputGate = Activation.Sigmoid<T>().apply(listOf(inputGate)).first()
            outputGate = Activation.Sigmoid<T>().apply(listOf(outputGate)).first()
            forgetGate = Activation.Sigmoid<T>().apply(listOf(forgetGate)).first()
            cellGate = Activation.Tanh<T>().apply(listOf(cellGate)).first()

            val newC = forgetGate.multiply(prevC) + inputGate.multiply(cellGate)
            val newH = outputGate.multiply(Activation.Tanh<T>().apply(listOf(newC!!)).first())

            prevH = newH
            prevC = newC
            resH.add(newH)
        }

        return listOf(prevH)
    }

    private fun addBiases(gates: Tensor<T>, biases: Tensor<T>, hiddenSize: Int) =
        gates.mapIndexed { index, t -> t + biases.data[intArrayOf(index[0], 0)] + biases.data[intArrayOf(index[0] + 4 * hiddenSize, 0)] }

    @Suppress("UNCHECKED_CAST")
    private fun getGates(gates: Tensor<T>?, batchSize: Long, hiddenSize: Long): List<Tensor<T>> {
        val chunkedBuffer = gates!!.data.buffer.asIterable().chunked((batchSize * hiddenSize).toInt())

        val shape = listOf(hiddenSize, batchSize)
        val mySpace = resolveSpaceWithKClass(gates.type!!.resolveKClass(), shape.toIntArray())
        mySpace as TensorRing<T>
        return List(chunkedBuffer.size) { Tensor(shape.reversed(), chunkedBuffer[it], gates.type, null, mySpace) }
    }
}

