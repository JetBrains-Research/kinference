package org.jetbrains.research.kotlin.mpp.inference.operators.layer.recurrent.lstm

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.mathExtension.*
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.*
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.Sigmoid
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.Tanh
import scientifik.kmath.structures.BufferNDStructure
import scientifik.kmath.structures.get

open class LSTMLayer<T : Number> {
    open fun apply(inputs: List<Tensor>): List<Tensor> {
        require(inputs.size in 3..4) { "Applicable only for three or four arguments" }

        val inputList = inputs.toList()

        val inputTensor = inputList[0]
        val weights = inputList[1].squeeze(0)
        val recWeights = inputList[2].squeeze(0)
        val bias = inputList.getOrNull(3)

        val batchSize = inputTensor.data.shape[1]
        val hiddenSize = recWeights.data.shape[1]

        val (mainOutput, currentState) = activate(inputTensor.as2DList(), weights, recWeights, bias)
        val shapeForOutput = intArrayOf(1, batchSize, hiddenSize)
        return listOf(mainOutput.toOutput(), currentState.output.reshape(shapeForOutput), currentState.cellGate.reshape(shapeForOutput))
    }

    protected fun activate(inputMatrices: Collection<Tensor>, weights: Tensor, recWeights: Tensor, bias: Tensor?): Pair<List<Tensor>, State> {
        val hiddenSize = recWeights.data.shape[1]
        val batchSize = inputMatrices.first().data.shape[0]

        var currentState = State.initialize(batchSize, hiddenSize, inputMatrices.first().info.type)
        val biasesData = if (bias != null) BiasesData.create(bias, hiddenSize, batchSize) else null
        val weightsTranspose = weights.transpose()
        val recWeightsTranspose = recWeights.transpose()

        val mainOutput = inputMatrices.map { inputMatrix ->
            val gatesData = GatesData.create(inputMatrix, weightsTranspose, recWeightsTranspose, currentState, biasesData)

            val activatedGatesData = gatesData.activate()

            currentState = State.create(activatedGatesData, currentState)

            currentState.output
        }

        return Pair(mainOutput, currentState)
    }

    data class GatesData(
        val inputGate: Tensor,
        val outputGate: Tensor,
        val forgetGate: Tensor,
        val cellGate: Tensor
    ) {
        val sigmoid = Sigmoid()
        val tanh = Tanh()

        fun activate(): GatesData {
            val activatedInputGate = sigmoid.activate(inputGate)
            val activatedOutputGate = sigmoid.activate(outputGate)
            val activatedForgetGate = sigmoid.activate(forgetGate)
            val activatedCellGate = tanh.activate(cellGate)
            return GatesData(activatedInputGate, activatedOutputGate, activatedForgetGate, activatedCellGate)
        }

//        private fun calcGates(tensor1: Tensor, tensor2: Tensor, tensor3: Tensor, activate: ((Number) -> Number)? = null): Tensor {
//            val (buffer, type) = if (activate == null) {
//                createInferredTypeBuffer(tensor1.info.type, tensor1.info.type, tensor1.data.strides.linearSize) {
//                    add(tensor1.data.buffer[it] as Number, tensor2.data.buffer[it] as Number, tensor3.data.buffer[it] as Number)
//                }
//            } else {
//                createInferredTypeBuffer(tensor1.info.type, tensor1.info.type, tensor1.data.strides.linearSize) {
//                    activate(add(tensor1.data.buffer[it] as Number, tensor2.data.buffer[it] as Number, tensor3.data.buffer[it] as Number))
//                }
//            }
//
//            return Tensor(null, BufferNDStructure(tensor1.data.strides, buffer as Buffer<Any>), type)
//        }

//        fun addBiases(weightsBiasesData: BiasesData, recursiveWeightsBiasesData: BiasesData, activation: Boolean = false): GatesData {
//            val inputGateWithBiases = calcGates(inputGate, weightsBiasesData.inputGateBiases, recursiveWeightsBiasesData.inputGateBiases, if (activation) (Sigmoid)::activate else null)
//            val outputGateWithBiases = calcGates(outputGate, weightsBiasesData.outputGateBiases, recursiveWeightsBiasesData.outputGateBiases, if (activation) (Sigmoid)::activate else null)
//            val forgetGateWithBiases = calcGates(forgetGate, weightsBiasesData.forgetGateBiases, recursiveWeightsBiasesData.forgetGateBiases, if (activation) (Sigmoid)::activate else null)
//            val cellGateWithBiases = calcGates(cellGate, weightsBiasesData.cellGateBiases, recursiveWeightsBiasesData.cellGateBiases, if (activation) (Tanh)::activate else null)
//
//            return GatesData(inputGateWithBiases, outputGateWithBiases, forgetGateWithBiases, cellGateWithBiases)
//        }


        companion object {
            fun create(inputMatrix: Tensor, weights: Tensor, recWeights: Tensor, prevState: State, bias: Tensor?): GatesData {
                val gates = (inputMatrix.matmul(weights) + prevState.output.matmul(recWeights)) as Tensor
                val gatesWithBias = if (bias != null) (gates + bias) as Tensor else gates
                val gatesList = gatesWithBias.splitHorizontal(4)
                return GatesData(gatesList[0], gatesList[1], gatesList[2], gatesList[3])
            }
        }
    }

    data class State(val output: Tensor, val cellGate: Tensor) {

        companion object {
            private val tanh = Tanh()

            @Suppress("UNCHECKED_CAST")
            fun initialize(batchSize: Int, hiddenSize: Int, type: TensorProto.DataType): State {
                val newShape = intArrayOf(batchSize, hiddenSize)
                val zeros = BufferNDStructure(TensorStrides(newShape), createBuffer(type, batchSize * hiddenSize) { 0.0f }) as BufferNDStructure<Any>
                return State(Tensor(null, zeros, type), Tensor(null, zeros, type))
            }

            fun create(gatesData: GatesData, prevState: State): State {
                val cellGateTensor = (gatesData.forgetGate * prevState.cellGate + gatesData.inputGate * gatesData.cellGate) as Tensor

                val outputTensor = (gatesData.outputGate * tanh.activate(cellGateTensor)) as Tensor

                return State(outputTensor, cellGateTensor)
            }
        }
    }

    data class BiasesData(
        val inputGateBiases: Tensor,
        val outputGateBiases: Tensor,
        val forgetGateBiases: Tensor,
        val cellGateBiases: Tensor
    ) {
        companion object {
            fun create(biases: Tensor, hiddenSize: Int, batchSize: Int): Tensor {
                val shape = intArrayOf(batchSize, 4 * hiddenSize)
                val newStrides = TensorStrides(shape)

                val leftBuffer = createBuffer(biases.info.type, newStrides.linearSize) { i ->
                    val (_, colNum) = newStrides.index(i)

                    biases.data.buffer[colNum]
                }
                val leftTensor = BufferNDStructure(newStrides, leftBuffer)

                val rightBuffer = createBuffer(biases.info.type, newStrides.linearSize) { i ->
                    val (_, colNum) = newStrides.index(i)

                    biases.data.buffer[colNum + shape[1]]
                }
                val rightTensor = BufferNDStructure(newStrides, rightBuffer)

                return Tensor("bias", leftTensor.plus(rightTensor), biases.info.type)


//                @Suppress("UNCHECKED_CAST")
//                val parsedBiases = List(8) { index ->
//                    val (buffer, _) = createInferredTypeBuffer(biases.info.type, biases.info.type, newStrides.linearSize) { i ->
//                        val indices = newStrides.index(i)
//                        val colNum = indices[1]
//                        biases.data.buffer[hiddenSize * index + colNum]
//                    }
//                    val newStructure = BufferNDStructure(newStrides, buffer)
//                    Tensor(null, newStructure, biases.info.type)
//                }
//
//                return BiasesData(
//                    (parsedBiases[0] + parsedBiases[4]) as Tensor,
//                    (parsedBiases[1] + parsedBiases[5]) as Tensor,
//                    (parsedBiases[2] + parsedBiases[6]) as Tensor,
//                    (parsedBiases[3] + parsedBiases[7]) as Tensor
//                )
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun List<Tensor>.toOutput(): Tensor {
        val newShape = intArrayOf(this.size, 1, this.first().data.shape[0], this.first().data.shape[1])
        val newStrides = TensorStrides(newShape)

        val type = this.first().info.type
        val (buffer, _) = createInferredTypeBuffer(type, type, newStrides.linearSize) { i ->
            val indices = newStrides.index(i)
            val (inputNum, _, rowNum, colNum) = indices
            this[inputNum].data[rowNum, colNum]
        }

        val newBuffer = BufferNDStructure(newStrides, buffer)
        return Tensor(null, newBuffer, type)
    }
}
