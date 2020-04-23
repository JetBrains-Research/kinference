package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.Activation
import org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.RecurrentLayer
import org.jetbrains.research.kotlin.mpp.inference.space.*
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.types.resolveKClass
import scientifik.kmath.structures.*

open class LSTMLayer<T : Number> : RecurrentLayer<T>() {
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
        val biasesData = if (bias != null) BiasesData.create(bias, hiddenSize, batchSize) else null

        return inputTensor.as2DCollection().map { inputMatrix ->
            val gatesData = GatesData.create(inputMatrix, weights, recWeights, currentState)

            val gatesDataWithBiases = if (biasesData != null) gatesData.addBiases(biasesData.first, biasesData.second) else gatesData
            val activatedGatesData = gatesDataWithBiases.activate()

            val newCellGate = activatedGatesData.forgetGate * currentState.cellGate + activatedGatesData.inputGate * activatedGatesData.cellGate
            val newOutput = activatedGatesData.outputGate * Activation.Tanh<T>().apply(newCellGate).first()

            currentState = State(newOutput, newCellGate)

            newOutput
        }.toOutput()
    }

    data class GatesData<T : Number>(val inputGate: Tensor<T>,
                                     val outputGate: Tensor<T>,
                                     val forgetGate: Tensor<T>,
                                     val cellGate: Tensor<T>) {
        fun activate() : GatesData<T> {
            val activatedInputGate = Activation.Sigmoid<T>().apply(inputGate).first()
            val activatedOutputGate = Activation.Sigmoid<T>().apply(outputGate).first()
            val activatedForgetGate = Activation.Sigmoid<T>().apply(forgetGate).first()
            val activatedCellGate = Activation.Tanh<T>().apply(cellGate).first()
            return GatesData(activatedInputGate, activatedOutputGate, activatedForgetGate, activatedCellGate)
        }

        fun addBiases(weightsBiasesData: BiasesData<T>, recursiveWeightsBiasesData: BiasesData<T>) : GatesData<T> {
            val inputGateWithBiases = inputGate + weightsBiasesData.inputGateBiases + recursiveWeightsBiasesData.inputGateBiases
            val outputGateWithBiases = outputGate + weightsBiasesData.outputGateBiases + recursiveWeightsBiasesData.outputGateBiases
            val forgetGateWithBiases = forgetGate + weightsBiasesData.forgetGateBiases + recursiveWeightsBiasesData.forgetGateBiases
            val cellGateWithBiases = cellGate + weightsBiasesData.cellGateBiases + recursiveWeightsBiasesData.forgetGateBiases

            return GatesData(inputGateWithBiases, outputGateWithBiases, forgetGateWithBiases, cellGateWithBiases)
        }

        companion object {
            fun <T : Number> create(inputMatrix: Tensor<T>, weights: Tensor<T>, recWeights: Tensor<T>,
                                    prevState: State<T>): GatesData<T> {
                val gates = inputMatrix.dot(weights.transpose()) + prevState.output.dot(recWeights.transpose())
                val gatesList = gates.splitByIndex(4, 1)
                return GatesData(gatesList[0], gatesList[1], gatesList[2], gatesList[3])
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

    data class BiasesData<T : Number>(val inputGateBiases: Tensor<T>,
                                      val outputGateBiases: Tensor<T>,
                                      val forgetGateBiases: Tensor<T>,
                                      val cellGateBiases: Tensor<T>) {
        companion object {
            fun <T : Number> create(biases: Tensor<T>, hiddenSize: Int, batchSize: Int) : Pair<BiasesData<T>, BiasesData<T>> {
                val shape = intArrayOf(hiddenSize, batchSize)
                val blockSize = hiddenSize * batchSize
                val newStrides = SpaceStrides(shape)
                @Suppress("UNCHECKED_CAST")
                val newSpace = resolveSpaceWithKClass(biases.type!!.resolveKClass(), shape) as TensorRing<T>
                val parsedBiases = List(8) { index ->
                    val newBuffer = VirtualBuffer(blockSize) { i ->
                        val indices = newStrides.index(i)
                        val rowNum = indices[0]
                        biases.data.buffer[hiddenSize * index + rowNum]
                    }
                    val newStructure = BufferNDStructure(newStrides, newBuffer)
                    Tensor(null, newStructure, biases.type, newSpace)
                }
                val weightsBiasesData = BiasesData(parsedBiases[0], parsedBiases[1], parsedBiases[2], parsedBiases[3])
                val recursiveWeightsBiasesData = BiasesData(parsedBiases[4], parsedBiases[5], parsedBiases[6], parsedBiases[7])
                return Pair(weightsBiasesData, recursiveWeightsBiasesData)
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : Number> List<Tensor<T>>.toOutput(): Collection<Tensor<T>> {
        val newShape = intArrayOf(this.size, 1, this.first().data.shape[0], this.first().data.shape[1])
        val newStrides = SpaceStrides(newShape)
        val newData = VirtualBuffer(newStrides.linearSize) { i ->
            val indices = newStrides.index(i)
            val inputNum = indices[0]
            val rowNum = indices[2]
            val colNum = indices[3]
            this[inputNum].data[rowNum, colNum]
        }
        val newBuffer = BufferNDStructure(newStrides, newData)
        val newSpace = resolveSpaceWithKClass(this.first().type!!.resolveKClass(), newShape) as TensorRing<T>
        return listOf(Tensor("Y", newBuffer, this.first().type, newSpace))
    }
}
