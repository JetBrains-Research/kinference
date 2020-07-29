package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.matrixTranspose
import org.jetbrains.research.kotlin.inference.extensions.ndarray.splitWithAxis
import org.jetbrains.research.kotlin.inference.extensions.primitives.matrixDotInto
import org.jetbrains.research.kotlin.inference.extensions.primitives.reversed
import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType
import org.jetbrains.research.kotlin.inference.operators.activations.Activation
import org.jetbrains.research.kotlin.inference.operators.layer.recurrent.RecurrentLayer

open class NewLSTM(hiddenSize: Int, activations: List<String>, direction: String) : RecurrentLayer(hiddenSize, activations, direction) {

    private var parsedWeights: GatesData? = null
    private var weights: Tensor? = null

    private var parsedRecurrentWeights: GatesData? = null
    private var recurrentWeights: Tensor? = null

    private var parsedBias: GatesData? = null
    private var bias: Tensor? = null

    private var parsedPeepholes: GatesData? = null
    private var peepholes: Tensor? = null

    private var parsedInitialOutput: List<NDArray<Any>>? = null
    private var initialOutput: Tensor? = null

    private var parsedInitialCellState: List<NDArray<Any>>? = null
    private var initialCellState: Tensor? = null

    private var seqLength: Int? = null
    private var batchSize: Int? = null

    private var type: DataType? = null

    init {
        require(direction == "forward" || direction == "reverse")
    }

    override fun apply(inputList: List<Tensor>): List<Tensor> {
        /*fun apply(input: Tensor, weights: Tensor, recurrentWeights: Tensor, bias: Tensor?, sequenceLens: Tensor?, initialOutput: Tensor?, initialCellState: Tensor?,
              peepholes: Tensor?): List<Tensor> {*/
        val input = inputList[0]
        val weights = inputList[1]
        val recurrentWeights = inputList[2]
        val bias = inputList.getOrNull(3)
        val sequenceLens = inputList.getOrNull(4)
        val initialOutput = inputList.getOrNull(5)
        val initialCellState = inputList.getOrNull(6)
        val peepholes = inputList.getOrNull(7)

        seqLength = input.data.shape[0]
        batchSize = input.data.shape[1]
        type = input.info.type

        parseWeights(weights)
        parseRecurrentWeights(recurrentWeights)
        parseBias(bias)
        parsePeepholes(peepholes)
        parseInitialOutput(initialOutput)
        parseInitialCellState(initialCellState)


        val (mainOutput, lastState) = activate(parseInput(input), parsedWeights!!, parsedRecurrentWeights!!, parsedBias, parseSeqLens(sequenceLens),
            parsedInitialOutput, parsedInitialCellState, parsedPeepholes, type!!, batchSize!!, seqLength!!)
        return listOf(parseOutput(mainOutput).asTensor(), lastState.first.asTensor(), lastState.second.asTensor())
    }

    fun activate(inputs: List<List<NDArray<Any>>>, weights: GatesData, recWeights: GatesData, bias: GatesData?, sequenceLens: IntArray,
                 initialOutput: List<NDArray<Any>>?, initialCellState: List<NDArray<Any>>?, peepholes: GatesData?, type: DataType, batchSize: Int, seqLength: Int):
        Pair<List<NDArray<Any>>, Pair<NDArray<Any>, NDArray<Any>>> {

        val gatesData = GatesData.allocateGates(hiddenSize, type)

        val (f, g, h) = activations.map { Activation.create(it, type) }

        val prevOutput = (initialOutput?.map { it.clone() } ?: List(batchSize) { allocateNDArray(type, gatesData.input.strides) }).toMutableList()
        var isPrevZero = initialOutput == null

        val prevCellState = (initialCellState?.map { it.clone() }) ?: List(batchSize) { allocateNDArray(type, gatesData.input.strides) }
        var isCellStateZero = initialCellState == null

        //val outputArray = allocateNDArray(type!!, Strides(intArrayOf(seqLength!!, 1, batchSize!!, hiddenSize)))
        val outputStrides = Strides(intArrayOf(1, batchSize, hiddenSize))

        val actualIndices = if (direction == "forward") inputs.indices.toList().toIntArray() else inputs.indices.reversed()

        val out = actualIndices.map { batchNum ->
            val outputArray = allocateNDArray(type, outputStrides)
            for ((i, input) in inputs[batchNum].withIndex()) {
                if (batchNum >= sequenceLens[i]) {
                    if (batchNum == sequenceLens[i]) prevOutput[i].clean()
                    continue
                }

                input.matrixDotInto(weights.input, gatesData.input, true)
                if (!isPrevZero) prevOutput[i].matrixDotInto(recWeights.input, gatesData.input, false)
                if (!isCellStateZero && peepholes != null) gatesData.input.plus(peepholes.input * prevCellState[i], false)
                if (bias != null) gatesData.input.plus(bias.input, false)
                gatesData.input.mapElements(f, false)

                input.matrixDotInto(weights.forget, gatesData.forget, true)
                if (!isPrevZero) prevOutput[i].matrixDotInto(recWeights.forget, gatesData.forget, false)
                if (!isCellStateZero && peepholes != null) gatesData.forget.plus(peepholes.forget * prevCellState[i], false)
                if (bias != null) gatesData.forget.plus(bias.forget, false)
                gatesData.forget.mapElements(f, false)

                input.matrixDotInto(weights.cellGate, gatesData.cellGate, true)
                if (!isPrevZero) prevOutput[i].matrixDotInto(recWeights.cellGate, gatesData.cellGate, false)
                if (bias != null) gatesData.cellGate.plus(bias.cellGate, false)
                gatesData.cellGate.mapElements(g, false)

                if (!isCellStateZero) prevCellState[i].times(gatesData.forget, false)
                prevCellState[i].plus(gatesData.input * gatesData.cellGate, false)

                input.matrixDotInto(weights.output, gatesData.output, true)
                if (!isPrevZero) prevOutput[i].matrixDotInto(recWeights.output, gatesData.output, false)
                if (peepholes != null) gatesData.output.plus(peepholes.output * prevCellState[i], false)
                if (bias != null) gatesData.output.plus(bias.output, false)
                gatesData.output.mapElements(f, false)

                prevOutput[i] = gatesData.output * prevCellState[i].mapElements(h, true)

                outputArray.placeAll(i * outputArray.strides.strides[1], prevOutput[i].array)
                //outputArray.placeAll(batchNum * tempStrides.strides[1] *  + i * outputArray.strides.strides[2], prevOutput[i].array)
            }
            isCellStateZero = false
            isPrevZero = false
            outputArray
        }

        val strides = Strides(intArrayOf(1, batchSize, hiddenSize))
        val cellStateArray = allocateNDArray(type, strides)

        for (i in prevOutput.indices) {
            cellStateArray.placeAll(strides.strides[1] * i, prevCellState[i].array)
        }

        val outputList = if (direction == "forward") out else out.asReversed()

        return Pair(outputList, Pair(out.last(), cellStateArray))
    }

    private fun parseWeights(weights: Tensor) {
        if (parsedWeights == null || this.weights !== weights) {
            this.parsedWeights = GatesData.createWeights(weights.data)
            this.weights = weights
        }
    }

    private fun parseRecurrentWeights(recWeights: Tensor) {
        if (parsedRecurrentWeights == null || this.recurrentWeights !== recWeights) {
            this.parsedRecurrentWeights = GatesData.createWeights(recWeights.data)
            this.recurrentWeights = recWeights
        }
    }

    private fun parseBias(bias: Tensor?) {
        if (bias != null && (parsedBias == null || this.bias !== bias)) {
            this.parsedBias = GatesData.createBias(bias.data)
            this.bias = bias
        }
    }

    private fun parsePeepholes(peepholes: Tensor?) {
        if (peepholes != null && (parsedPeepholes == null || this.peepholes !== peepholes)) {
            this.parsedPeepholes = GatesData.createPeepholes(peepholes.data)
            this.peepholes = peepholes
        }
    }

    private fun parseInitialOutput(initialOutput: Tensor?) {
        if (initialOutput != null && (parsedInitialOutput == null || this.initialOutput !== initialOutput)) {
            this.parsedInitialOutput = initialOutput.data.squeeze(0).splitWithAxis(batchSize!!)
            this.initialOutput = initialOutput
        }
    }

    private fun parseInitialCellState(initialCellState: Tensor?) {
        if (initialCellState != null && (parsedInitialCellState == null || this.initialCellState !== initialCellState)) {
            this.parsedInitialCellState = initialCellState.data.squeeze(0).splitWithAxis(batchSize!!)
            this.initialCellState = initialCellState
        }
    }

    private fun parseInput(input: Tensor): List<List<NDArray<Any>>> =
        input.data.splitWithAxis(seqLength!!, 0, false).map { it.splitWithAxis(batchSize!!, 0, true) }

    private fun parseSeqLens(sequenceLens: Tensor?): IntArray = sequenceLens?.data?.array as? IntArray ?: IntArray(batchSize!!) { seqLength!! }

    private fun parseOutput(output: List<NDArray<Any>>): NDArray<Any> {
        val newShape = intArrayOf(seqLength!!, 1, batchSize!!, hiddenSize)
        val newStrides = Strides(newShape)
        val outputArray = allocateNDArray(type!!, newStrides)

        for (i in output.indices) {
            outputArray.placeAll(i * newStrides.strides[1], output[i].array)
        }

        return outputArray
    }

    data class GatesData(val input: NDArray<Any>, val output: NDArray<Any>, val forget: NDArray<Any>, val cellGate: NDArray<Any>) {
        companion object {
            fun createWeights(weights: NDArray<Any>): GatesData {
                require(weights.shape[0] == 1)

                val matrix = weights.squeeze(0)

                val weightsList = matrix.splitWithAxis(4)
                return GatesData(weightsList[0].matrixTranspose(), weightsList[1].matrixTranspose(),
                    weightsList[2].matrixTranspose(), weightsList[3].matrixTranspose())
            }

            fun createBias(bias: NDArray<Any>): GatesData {
                require(bias.shape[0] == 1)

                val linear = bias.squeeze(0)

                val biasList = linear.splitWithAxis(8)

                return GatesData(
                    biasList[0].plus(biasList[4]),
                    biasList[1].plus(biasList[5]),
                    biasList[2].plus(biasList[6]),
                    biasList[3].plus(biasList[7])
                )

                /*val newShape = intArrayOf(1, hiddenSize)
                val newStrides = Strides(newShape)

                val biasArray = when (bias.type){
                    TensorProto.DataType.FLOAT -> {
                        val array = bias.array as FloatArray

                        Array(4) {
                            val newArray = FloatArray(hiddenSize)
                            for (j in (0 until hiddenSize)){
                                newArray[j] = array[j + hiddenSize * it] + array[j + hiddenSize * (it + 4)]
                            }
                            FloatNDArray(newArray, newStrides)
                        }
                    }
                    TensorProto.DataType.DOUBLE -> {
                        val array = bias.array as DoubleArray

                        Array(4) {
                            val newArray = DoubleArray(hiddenSize)
                            for (j in (0 until hiddenSize)){
                                newArray[j] = array[j + hiddenSize * it] + array[j + hiddenSize * (it + 4)]
                            }
                            DoubleNDArray(newArray, newStrides)
                        }
                    }
                    else -> throw UnsupportedOperationException()
                }
                return GatesData(biasArray[0], biasArray[1], biasArray[2], biasArray[3])*/


            }

            fun createPeepholes(peepholes: NDArray<Any>): GatesData {
                require(peepholes.shape[0] == 1)

                val linear = peepholes.squeeze(0)

                val peepholesList = linear.splitWithAxis(3)
                return GatesData(peepholesList[0], peepholesList[1], peepholesList[2], allocateNDArray(DataType.INT16, Strides(intArrayOf(1))))
            }

            fun allocateGates(hiddenSize: Int, type: DataType): GatesData {
                val newStrides = Strides(intArrayOf(1, hiddenSize))

                val allocArrays = Array(4) {
                    allocateNDArray(type, newStrides)
                }
                return GatesData(allocArrays[0], allocArrays[1], allocArrays[2], allocArrays[3])
            }
        }
    }
}
