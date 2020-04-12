package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.Activation
import org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.RecurrentLayer
import org.jetbrains.research.kotlin.mpp.inference.space.SpaceStrides
import org.jetbrains.research.kotlin.mpp.inference.space.TensorRing
import org.jetbrains.research.kotlin.mpp.inference.space.resolveSpaceWithKClass
import org.jetbrains.research.kotlin.mpp.inference.space.toIntArray
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.types.resolveKClass
import scientifik.kmath.structures.BufferNDStructure
import scientifik.kmath.structures.asBuffer
import scientifik.kmath.structures.asIterable

class LSTMLayer<T : Number> : RecurrentLayer<T>() {
    @Suppress("UNCHECKED_CAST")
    override fun apply(inputs: Collection<Tensor<T>>): Collection<Tensor<T>> {
        require(inputs.size in 3..4) { "Applicable only for three or four arguments" }

        val inputList = inputs.toList()

        val inputTensor = inputList[0]
        val weights = inputList[1].as2DCollection().first()
        val recWeights = inputList[2].as2DCollection().first()
        val bias = inputList.getOrNull(3)

        val hiddenSize = recWeights.data.shape[1]
        val batchSize = inputTensor.data.shape[1]

        var currentState = State.initialize<T>(hiddenSize, batchSize, inputTensor.type!!)

        return inputTensor.as2DCollection().map { inputMatrix ->
            val gatesData = GatesData.create(inputMatrix, weights, recWeights, currentState)
            val gatesDataWithBiases = if (bias != null) gatesData.addBiases(bias, hiddenSize) else gatesData
            val activatedGatesData = gatesDataWithBiases.activate()

            val newCellGate = activatedGatesData.forgetGate.multiply(currentState.cellGate) + activatedGatesData.inputGate.multiply(activatedGatesData.cellGate)
            val newOutput = activatedGatesData.outputGate.multiply(Activation.Tanh<T>().apply(listOf(newCellGate)).first())

            currentState = State(newOutput, newCellGate)

            newOutput.transpose()
        }.toOutput()
    }

    data class GatesData<T : Number>(var inputGate: Tensor<T>,
                                     var outputGate: Tensor<T>,
                                     var forgetGate: Tensor<T>,
                                     var cellGate: Tensor<T>) {
        fun activate() : GatesData<T> {
            val activatedInputGate = Activation.Sigmoid<T>().apply(listOf(inputGate)).first()
            val activatedOutputGate = Activation.Sigmoid<T>().apply(listOf(outputGate)).first()
            val activatedForgetGate = Activation.Sigmoid<T>().apply(listOf(forgetGate)).first()
            val activatedCellGate = Activation.Tanh<T>().apply(listOf(cellGate)).first()
            return GatesData(activatedInputGate, activatedOutputGate, activatedForgetGate, activatedCellGate)
        }

        fun addBiases(biases : Tensor<T>, hiddenSize: Int) : GatesData<T> {
            val chunkedBiasesBuffer = biases.data.buffer.asIterable().chunked(hiddenSize)

            val inputGateWithBiases = inputGate.mapIndexed() { index, t -> t + chunkedBiasesBuffer[0][index[0]] + chunkedBiasesBuffer[4][index[0]] }
            val outputGateWithBiases = outputGate.mapIndexed() { index, t -> t + chunkedBiasesBuffer[1][index[0]] + chunkedBiasesBuffer[5][index[0]] }
            val forgetGateWithBiases = forgetGate.mapIndexed { index, t -> t + chunkedBiasesBuffer[2][index[0]] + chunkedBiasesBuffer[6][index[0]] }
            val cellGateWithBiases = cellGate.mapIndexed { index, t -> t + chunkedBiasesBuffer[3][index[0]] + chunkedBiasesBuffer[7][index[0]] }
            return GatesData(inputGateWithBiases, outputGateWithBiases, forgetGateWithBiases, cellGateWithBiases)
        }

        companion object {
            @Suppress("UNCHECKED_CAST")
            fun <T : Number> create(inputMatrix: Tensor<T>, weights: Tensor<T>, recWeights: Tensor<T>,
                                    prevState: State<T>): GatesData<T> {
                val gates = weights.dot(inputMatrix.transpose()) + recWeights.dot(prevState.output.transpose())
                val hiddenSize = recWeights.data.shape[1]
                val batchSize = inputMatrix.data.shape[0]

                val chunkedGatesBuffer = gates.data.buffer.asIterable().chunked(batchSize * hiddenSize)

                val shape = listOf(hiddenSize.toLong(), batchSize.toLong())
                val gateSpace = resolveSpaceWithKClass(gates.type!!.resolveKClass(), shape.toIntArray()) as TensorRing<T>
                val res = List(chunkedGatesBuffer.size) { Tensor(shape, chunkedGatesBuffer[it], gates.type, null, gateSpace) }
                return GatesData(res[0], res[1], res[2], res[3])
            }
        }
    }

    data class State<T : Number>(val output: Tensor<T>, val cellGate: Tensor<T>) {
        companion object {
            @Suppress("UNCHECKED_CAST")
            fun <T : Number> initialize(batchSize: Int, hiddenSize: Int, type: TensorProto.DataType): State<T> {
                val stateSpace = resolveSpaceWithKClass(type.resolveKClass(), intArrayOf(batchSize, hiddenSize)) as TensorRing<T>
                return State(Tensor(null, stateSpace.zero, type, stateSpace), Tensor(null, stateSpace.zero, type, stateSpace))
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : Number> Collection<Tensor<T>>.toOutput(): Collection<Tensor<T>> {
        val newShape = intArrayOf(this.size, 1, this.first().data.shape[0], this.first().data.shape[1])
        val newData = this.flatMap { output -> output.data.buffer.asIterable() }
        val newBuffer = BufferNDStructure(SpaceStrides(newShape), newData.asBuffer())
        val newSpace = resolveSpaceWithKClass(this.first().type!!.resolveKClass(), newShape) as TensorRing<T>
        return listOf(Tensor("Y", newBuffer, this.first().type, newSpace))
    }
}
