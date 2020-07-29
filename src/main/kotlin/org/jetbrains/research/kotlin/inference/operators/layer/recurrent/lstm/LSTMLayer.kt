/*
package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.activations.Sigmoid
import org.jetbrains.research.kotlin.inference.operators.activations.Tanh

open class LSTMLayer<T : Number> {
    open fun apply(inputs: List<Tensor>): List<Tensor> {
        require(inputs.size in 3..4) { "Applicable only for three or four arguments" }

        val inputList = inputs.toList()

        val inputTensor = inputList[0]
        val weights = inputList[1].data.squeeze(0)
        val recWeights = inputList[2].data.squeeze(0)
        val bias = inputList.getOrNull(3)?.data

        val batchSize = inputTensor.data.shape[1]
        val hiddenSize = recWeights.shape[1]

        val (mainOutput, currentState) = activate(inputTensor.data.as2DList(), weights, recWeights, bias)
        val shapeForOutput = intArrayOf(1, batchSize, hiddenSize)
        return listOf(mainOutput.toOutput(), currentState.output.reshape(shapeForOutput).asTensor(), currentState.cellGate.reshape(shapeForOutput).asTensor())
    }

    protected fun activate(inputMatrices: Collection<NDArray<Any>>, weights: NDArray<Any>, recWeights: NDArray<Any>, bias: NDArray<Any>?): Pair<List<NDArray<Any>>, State> {
        val hiddenSize = recWeights.shape[1]
        val batchSize = inputMatrices.first().shape[0]

        var currentState = State.initialize(batchSize, hiddenSize, inputMatrices.first().type)
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
        val inputGate: NDArray<Any>,
        val outputGate: NDArray<Any>,
        val forgetGate: NDArray<Any>,
        val cellGate: NDArray<Any>
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
            fun create(inputMatrix: NDArray<Any>, weights: NDArray<Any>, recWeights: NDArray<Any>, prevState: State, bias: NDArray<Any>?): GatesData {
                val gates = (inputMatrix.matmul(weights).plus(prevState.output.matmul(recWeights)))
                val gatesWithBias = if (bias != null) (gates.plus(bias)) else gates
                val gatesList = gatesWithBias.splitHorizontal(4)
                return GatesData(gatesList[0], gatesList[1], gatesList[2], gatesList[3])
            }
        }
    }

    data class State(val output: NDArray<Any>, val cellGate: NDArray<Any>) {

        companion object {
            private val tanh = Tanh()

            @Suppress("UNCHECKED_CAST")
            fun initialize(batchSize: Int, hiddenSize: Int, type: TensorProto.DataType): State {
                val newShape = intArrayOf(batchSize, hiddenSize)
                val zeros = allocateNDArray(type, Strides(newShape))
                return State(zeros, zeros)
            }

            fun create(gatesData: GatesData, prevState: State): State {
                val cellGateTensor = (gatesData.forgetGate * prevState.cellGate + gatesData.inputGate * gatesData.cellGate)

                val outputTensor = (gatesData.outputGate * tanh.activate(cellGateTensor))

                return State(outputTensor, cellGateTensor)
            }
        }
    }

    data class BiasesData(
        val inputGateBiases: NDArray<Any>,
        val outputGateBiases: NDArray<Any>,
        val forgetGateBiases: NDArray<Any>,
        val cellGateBiases: NDArray<Any>
    ) {
        companion object {
            fun create(biases: NDArray<Any>, hiddenSize: Int, batchSize: Int): NDArray<Any> {
                val shape = intArrayOf(batchSize, 4 * hiddenSize)
                val newStrides = Strides(shape)

                val leftTensor = createNDArray<Any>(biases.type, newStrides) { i ->
                    val (_, colNum) = newStrides.index(i)
                    biases[colNum]
                }

                val rightTensor = createNDArray<Any>(biases.type, newStrides) { i ->
                    val (_, colNum) = newStrides.index(i)
                    biases[colNum + shape[1]]
                }

                return leftTensor + rightTensor


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
    private fun List<NDArray<Any>>.toOutput(): Tensor {
        val newShape = intArrayOf(this.size, 1, this.first().shape[0], this.first().shape[1])
        val newStrides = Strides(newShape)

        val type = this.first().type
        return createNDArray<Any>(type, newStrides) { i ->
            val indices = newStrides.index(i)
            val (inputNum, _, rowNum, colNum) = indices
            this[inputNum].get(intArrayOf(rowNum, colNum))
        }.asTensor()
    }
}
*/
