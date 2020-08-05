package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.splitWithAxis

class BiLSTMLayer(hiddenSize: Int, activations: List<String>, direction: String) : LSTMBase(hiddenSize, activations, direction) {
    init {
        require(direction == "bidirectional")
        require(activations.size == 6)
    }

    var forwardLstmData: LSTMData? = null
    var reverseLstmData: LSTMData? = null

    override fun apply(inputs: List<List<NDArray<Any>>>, sequenceLens: IntArray, outputArray: NDArray<Any>, startOffset: Int): List<Tensor> {
        val forwardLayer = LSTMLayer.create(hiddenSize, activations.subList(0, 3), "forward", forwardLstmData!!, seqLength!!, batchSize!!, type!!)
        val reverseLayer = LSTMLayer.create(hiddenSize, activations.subList(3, 6), "reverse", reverseLstmData!!, seqLength!!, batchSize!!, type!!)

        val (_, forwardLastOutput, forwardLastCellState) = forwardLayer.apply(inputs, sequenceLens, outputArray, startOffset)
        val (output, reverseLastOutput, reverseLastCellState) = reverseLayer.apply(inputs, sequenceLens, outputArray, startOffset + batchSize!! * hiddenSize)

        return listOf(output, concatLasts(forwardLastOutput, reverseLastOutput).asTensor(), concatLasts(forwardLastCellState, reverseLastCellState).asTensor())
    }

    private fun concatLasts(forward: Tensor, reverse: Tensor): NDArray<Any> {
        val newShape = forward.data.shape.copyOf()
        newShape[0] = 2
        val newStrides = Strides(newShape)
        val newArray = allocateNDArray(type!!, newStrides)
        newArray.placeAll(0, forward.data.array)
        newArray.placeAll(forward.data.linearSize, reverse.data.array)
        return newArray
    }

    override fun parseTempInputs(weights: Tensor, recurrentWeights: Tensor, bias: Tensor?, initialOutput: Tensor?, initialCellState: Tensor?, peepholes: Tensor?) {
        if (forwardLstmData == null || reverseLstmData == null) {
            val (forwardParsedWeights, reverseParsedWeights) = weights.data.splitWithAxis(2).map { GatesData.createWeights(it) }
            val (forwardParsedRecWeights, reverseParsedRecWeights) = recurrentWeights.data.splitWithAxis(2).map { GatesData.createWeights(it) }
            forwardLstmData = LSTMData(forwardParsedWeights, forwardParsedRecWeights, null, null, null, null, type!!)
            reverseLstmData = LSTMData(reverseParsedWeights, reverseParsedRecWeights, null, null, null, null, type!!)

            this.weights = weights.data
            this.recurrentWeights = recurrentWeights.data
            this.bias = null
            this.initialOutput = null
            this.initialCellState = null
            this.peepholes = null
        }
        if (weights.data !== this.weights) {
            val (forward, reverse) = weights.data.splitWithAxis(2)
            forwardLstmData = forwardLstmData!!.updateWeights(GatesData.createWeights(forward))
            reverseLstmData = reverseLstmData!!.updateWeights(GatesData.createWeights(reverse))
            this.weights = weights.data
        }
        if (recurrentWeights.data !== this.recurrentWeights) {
            val (forward, reverse) = recurrentWeights.data.splitWithAxis(2)
            forwardLstmData = forwardLstmData!!.updateRecurrentWeights(GatesData.createWeights(forward))
            reverseLstmData = reverseLstmData!!.updateRecurrentWeights(GatesData.createWeights(reverse))
            this.recurrentWeights = recurrentWeights.data
        }
        if (bias != null && bias.data !== this.bias) {
            val (forward, reverse) = bias.data.splitWithAxis(2)
            forwardLstmData = forwardLstmData!!.updateBias(GatesData.createBias(forward))
            reverseLstmData = reverseLstmData!!.updateBias(GatesData.createBias(reverse))
            this.bias = bias.data
        }
        if (initialOutput != null && initialOutput.data !== this.initialOutput) {
            val (forward, reverse) = initialOutput.data.splitWithAxis(2)
            forwardLstmData = forwardLstmData!!.updateInitialOutput(forward.squeeze(0).splitWithAxis(batchSize!!))
            reverseLstmData = reverseLstmData!!.updateInitialOutput(reverse.squeeze(0).splitWithAxis(batchSize!!))
            this.initialOutput = initialOutput.data
        }
        if (initialCellState != null && initialCellState.data !== this.initialCellState) {
            val (forward, reverse) = initialCellState.data.splitWithAxis(2)
            forwardLstmData = forwardLstmData!!.updateInitialCellGate(forward.squeeze(0).splitWithAxis(batchSize!!))
            reverseLstmData = reverseLstmData!!.updateInitialCellGate(reverse.squeeze(0).splitWithAxis(batchSize!!))
            this.initialCellState = initialCellState.data
        }
        if (peepholes != null && peepholes.data !== this.peepholes) {
            val (forward, reverse) = peepholes.data.splitWithAxis(2)
            forwardLstmData = forwardLstmData!!.updatePeepholes(GatesData.createPeepholes(forward))
            reverseLstmData = reverseLstmData!!.updatePeepholes(GatesData.createPeepholes(reverse))
            this.peepholes = peepholes.data
        }
    }
}
