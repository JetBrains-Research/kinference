package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.Sigmoid
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.Tanh
import org.jetbrains.research.kotlin.mpp.inference.space.SpaceStrides
import org.jetbrains.research.kotlin.mpp.inference.space.TensorRing
import org.jetbrains.research.kotlin.mpp.inference.space.resolveSpaceWithKClass
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.types.resolveKClass
import scientifik.kmath.structures.BufferNDStructure
import scientifik.kmath.structures.VirtualBuffer
import scientifik.kmath.structures.get

open class LSTMLayer<T : Number> {
    open fun apply(inputs: Collection<Tensor>): Collection<Tensor> {
        require(inputs.size in 3..4) { "Applicable only for three or four arguments" }

        val inputList = inputs.toList()

        val inputTensor = inputList[0]
        val weights = inputList[1].squeeze(0)
        val recWeights = inputList[2].squeeze(0)
        val bias = inputList.getOrNull(3)

        val batchSize = inputTensor.data.shape[1]
        val hiddenSize = recWeights.data.shape[1]

        val (mainOutput, currentState) = activate(inputTensor.as2DCollection(), weights, recWeights, bias)
        val shapeForOutput = intArrayOf(1, batchSize, hiddenSize)
        return listOf(mainOutput.toOutput(), currentState.output.reshape(shapeForOutput), currentState.cellGate.reshape(shapeForOutput))
    }

    protected fun activate(inputMatrices: Collection<Tensor>, weights: Tensor, recWeights: Tensor, bias: Tensor?): Pair<List<Tensor>, State<T>> {
        val hiddenSize = recWeights.data.shape[1]
        val batchSize = inputMatrices.first().data.shape[0]

        var currentState = State.initialize<T>(batchSize, hiddenSize, inputMatrices.first().type!!)
        val biasesData = if (bias != null) BiasesData.create<T>(bias, hiddenSize, batchSize) else null

        val mainOutput = inputMatrices.map { inputMatrix ->
            val gatesData = GatesData.create(inputMatrix, weights, recWeights, currentState)

            val gatesDataWithBiases = if (biasesData != null) gatesData.addBiases(biasesData.first, biasesData.second) else gatesData
            val activatedGatesData = gatesDataWithBiases.activate()

            currentState = State.create(activatedGatesData, currentState)

            currentState.output
        }

        return Pair(mainOutput, currentState)
    }

    data class GatesData<T : Number>(val inputGate: Tensor,
                                     val outputGate: Tensor,
                                     val forgetGate: Tensor,
                                     val cellGate: Tensor) {
        fun activate(): GatesData<T> {
            val activatedInputGate = Sigmoid().apply(inputGate).first()
            val activatedOutputGate = Sigmoid().apply(outputGate).first()
            val activatedForgetGate = Sigmoid().apply(forgetGate).first()
            val activatedCellGate = Tanh().apply(cellGate).first()
            return GatesData(activatedInputGate, activatedOutputGate, activatedForgetGate, activatedCellGate)
        }

        fun addBiases(weightsBiasesData: BiasesData<T>, recursiveWeightsBiasesData: BiasesData<T>): GatesData<T> {
            val inputGateWithBiases = inputGate + weightsBiasesData.inputGateBiases + recursiveWeightsBiasesData.inputGateBiases
            val outputGateWithBiases = outputGate + weightsBiasesData.outputGateBiases + recursiveWeightsBiasesData.outputGateBiases
            val forgetGateWithBiases = forgetGate + weightsBiasesData.forgetGateBiases + recursiveWeightsBiasesData.forgetGateBiases
            val cellGateWithBiases = cellGate + weightsBiasesData.cellGateBiases + recursiveWeightsBiasesData.forgetGateBiases

            return GatesData(inputGateWithBiases, outputGateWithBiases, forgetGateWithBiases, cellGateWithBiases)
        }

        companion object {
            fun <T : Number> create(inputMatrix: Tensor, weights: Tensor, recWeights: Tensor,
                                    prevState: State<T>): GatesData<T> {
                val gates = inputMatrix.dot(weights.transpose()) + prevState.output.dot(recWeights.transpose())
                val gatesList = gates.splitByIndex(4, 1)
                return GatesData(gatesList[0], gatesList[1], gatesList[2], gatesList[3])
            }
        }
    }

    data class State<T : Number>(val output: Tensor, val cellGate: Tensor) {
        companion object {
            @Suppress("UNCHECKED_CAST")
            fun <T : Number> initialize(batchSize: Int, hiddenSize: Int, type: TensorProto.DataType): State<T> {
                val stateSpace = resolveSpaceWithKClass(type.resolveKClass(), intArrayOf(batchSize, hiddenSize)) as TensorRing<Any>
                return State(Tensor(null, stateSpace.zero, type, stateSpace), Tensor(null, stateSpace.zero, type, stateSpace))
            }

            fun <T : Number> create(gatesData: GatesData<T>, prevState: State<T>): State<T> {
                val newCellGate = gatesData.forgetGate * prevState.cellGate + gatesData.inputGate * gatesData.cellGate
                val newOutput = gatesData.outputGate * Tanh().apply(newCellGate).first()
                return State(newOutput, newCellGate)
            }
        }
    }

    data class BiasesData<T : Number>(val inputGateBiases: Tensor,
                                      val outputGateBiases: Tensor,
                                      val forgetGateBiases: Tensor,
                                      val cellGateBiases: Tensor) {
        companion object {
            fun <T : Number> create(biases: Tensor, hiddenSize: Int, batchSize: Int): Pair<BiasesData<T>, BiasesData<T>> {
                val shape = intArrayOf(batchSize, hiddenSize)
                val blockSize = hiddenSize * batchSize
                val newStrides = SpaceStrides(shape)

                @Suppress("UNCHECKED_CAST")
                val newSpace = resolveSpaceWithKClass(biases.type!!.resolveKClass(), shape) as TensorRing<Any>
                val parsedBiases = List(8) { index ->
                    val newBuffer = VirtualBuffer(blockSize) { i ->
                        val indices = newStrides.index(i)
                        val rowNum = indices[0]
                        biases.data.buffer[hiddenSize * index + rowNum]
                    }
                    val newStructure = BufferNDStructure(newStrides, newBuffer)
                    Tensor(null, newStructure, biases.type, newSpace)
                }
                val weightsBiasesData = BiasesData<T>(parsedBiases[0], parsedBiases[1], parsedBiases[2], parsedBiases[3])
                val recursiveWeightsBiasesData = BiasesData<T>(parsedBiases[4], parsedBiases[5], parsedBiases[6], parsedBiases[7])
                return Pair(weightsBiasesData, recursiveWeightsBiasesData)
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun List<Tensor>.toOutput(): Tensor {
        val newShape = intArrayOf(this.size, 1, this.first().data.shape[0], this.first().data.shape[1])
        val newStrides = SpaceStrides(newShape)
        val newData = VirtualBuffer(newStrides.linearSize) { i ->
            val indices = newStrides.index(i)
            val (inputNum, _, rowNum, colNum) = indices
            this[inputNum].data[rowNum, colNum]
        }
        val newBuffer = BufferNDStructure(newStrides, newData)
        val newSpace = resolveSpaceWithKClass(this.first().type!!.resolveKClass(), newShape) as TensorRing<Any>
        return Tensor(null, newBuffer, this.first().type, newSpace)
    }
}
