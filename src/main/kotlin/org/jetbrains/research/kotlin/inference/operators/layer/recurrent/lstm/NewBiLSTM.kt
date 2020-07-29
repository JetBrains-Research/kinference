package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.splitWithAxis
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.layer.recurrent.RecurrentLayer
import org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm.NewLSTM.GatesData

class NewBiLSTM(hiddenSize: Int, activations: List<String>, direction: String) : RecurrentLayer(hiddenSize, activations, direction) {
    init {
        require(direction == "bidirectional")
        require(activations.size == 6)
    }

    val forwardLayer = NewLSTM(hiddenSize, activations.subList(0, 3), "forward")
    val reverseLayer = NewLSTM(hiddenSize, activations.subList(3, 6), "reverse")

    private var weights: Tensor? = null
    private var parsedWeights: Pair<GatesData, GatesData>? = null

    private var recurrentWeights: Tensor? = null
    private var parsedRecurrentWeights: Pair<GatesData, GatesData>? = null

    private var bias: Tensor? = null
    private var parsedBias: Pair<GatesData, GatesData>? = null

    private var peepholes: Tensor? = null
    private var parsedPeepholes: Pair<GatesData, GatesData>? = null

    private var initialOutput: Tensor? = null
    private var parsedInitialOutput: Pair<List<NDArray<Any>>, List<NDArray<Any>>>? = null

    private var initialCellState: Tensor? = null
    private var parsedInitialCellState: Pair<List<NDArray<Any>>, List<NDArray<Any>>>? = null

    private var seqLength: Int? = null
    private var batchSize: Int? = null

    private var type: TensorProto.DataType? = null

    override fun apply(inputList: List<Tensor>): List<Tensor> {
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

        val (forwardOutput, forwardState) = forwardLayer.activate(parseInput(input), parsedWeights!!.first, parsedRecurrentWeights!!.first, parsedBias?.first,
            parseSeqLens(sequenceLens), parsedInitialOutput?.first, parsedInitialCellState?.first, parsedPeepholes?.first, type!!, batchSize!!, seqLength!!)
        val (reverseOutput, reverseState) = reverseLayer.activate(parseInput(input), parsedWeights!!.second, parsedRecurrentWeights!!.second, parsedBias?.second,
            parseSeqLens(sequenceLens), parsedInitialOutput?.second, parsedInitialCellState?.second, parsedPeepholes?.second, type!!, batchSize!!, seqLength!!)

        return listOf(parseOutput(forwardOutput, reverseOutput).asTensor(),
            parseState(forwardState.first, reverseState.first).asTensor(),
            parseState(forwardState.second, reverseState.second).asTensor())
    }

    private fun parseWeights(weights: Tensor) {
        if (parsedWeights == null || weights !== this.weights) {
            val (forward, reverse) = weights.data.splitWithAxis(2)
            parsedWeights = Pair(GatesData.createWeights(forward), GatesData.createWeights(reverse))
        }
    }

    private fun parseRecurrentWeights(recurrentWeights: Tensor) {
        if (parsedRecurrentWeights == null || recurrentWeights !== this.recurrentWeights) {
            val (forward, reverse) = recurrentWeights.data.splitWithAxis(2)
            parsedRecurrentWeights = Pair(GatesData.createWeights(forward), GatesData.createWeights(reverse))
        }
    }

    private fun parseBias(bias: Tensor?) {
        if (bias != null && (parsedBias == null || bias !== this.bias)) {
            val (forward, reverse) = bias.data.splitWithAxis(2)
            parsedBias = Pair(GatesData.createBias(forward), GatesData.createBias(reverse))
        }
    }

    private fun parsePeepholes(peepholes: Tensor?) {
        if (peepholes != null && (parsedPeepholes == null || peepholes !== this.peepholes)) {
            val (forward, reverse) = peepholes.data.splitWithAxis(2)
            parsedPeepholes = Pair(GatesData.createPeepholes(forward), GatesData.createPeepholes(reverse))
        }
    }

    private fun parseInitialOutput(initialOutput: Tensor?) {
        if (initialOutput != null && (parsedInitialOutput == null || initialOutput !== this.initialOutput)) {
            val (forward, reverse) = initialOutput.data.splitWithAxis(2).map { it.squeeze(0).splitWithAxis(batchSize!!) }
            parsedInitialOutput = Pair(forward, reverse)
        }
    }

    private fun parseInitialCellState(initialCellState: Tensor?) {
        if (initialCellState != null && (parsedInitialCellState == null || initialCellState !== this.initialCellState)) {
            val (forward, reverse) = initialCellState.data.splitWithAxis(2).map { it.squeeze(0).splitWithAxis(batchSize!!) }
            parsedInitialCellState = Pair(forward, reverse)
        }
    }

    private fun parseInput(input: Tensor): List<List<NDArray<Any>>> =
        input.data.splitWithAxis(seqLength!!, 0, false).map { it.splitWithAxis(batchSize!!, 0, true) }

    private fun parseSeqLens(sequenceLens: Tensor?): IntArray = sequenceLens?.data?.array as? IntArray ?: IntArray(batchSize!!) { seqLength!! }

    private fun parseOutput(forwardOutput: List<NDArray<Any>>, reverseOutput: List<NDArray<Any>>): NDArray<Any> {
        val newShape = intArrayOf(seqLength!!, 2, batchSize!!, hiddenSize)
        val newStrides = Strides(newShape)
        val outputArray = allocateNDArray(type!!, newStrides)

        for (i in forwardOutput.indices) {
            outputArray.placeAll(newStrides.strides[0] * i, forwardOutput[i].array)
            outputArray.placeAll(newStrides.strides[0] * i + forwardOutput[i].linearSize, reverseOutput[i].array)
        }

        return outputArray
    }

    private fun parseState(forwardState: NDArray<Any>, reverseState: NDArray<Any>): NDArray<Any> {
        val newShape = intArrayOf(2, batchSize!!, hiddenSize)
        val newStrides = Strides(newShape)
        val outputState = allocateNDArray(type!!, newStrides)

        outputState.placeAll(0, forwardState.array)
        outputState.placeAll(forwardState.linearSize, reverseState.array)

        return outputState
    }
}
