package io.kinference.tfjs.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.tfjs.operators.layer.recurrent.LayerDirection

class BiLSTMLayer(hiddenSize: Int, activations: List<String>): LSTMLayerBase(hiddenSize, activations, LayerDirection.BIDIRECTIONAL) {
    init {
        require(activations.size == 6) { "Required number of activations is 6, but ${activations.size} found" }
    }

    private val forwardLayer = LSTMLayer(hiddenSize, activations.subList(0, 3), LayerDirection.FORWARD)
    private val reverseLayer = LSTMLayer(hiddenSize, activations.subList(3, 6), LayerDirection.REVERSE)

    override suspend fun apply(
        input: NumberNDArrayTFJS,
        weights: NumberNDArrayTFJS,
        recurrentWeights: NumberNDArrayTFJS,
        bias: NumberNDArrayTFJS?,
        sequenceLens: NumberNDArrayTFJS?,
        initialHiddenState: NumberNDArrayTFJS?,
        initialCellState: NumberNDArrayTFJS?,
        peepholes: NumberNDArrayTFJS?
    ): LSTMLayerOutput {
        val (outputArray, hiddenStateArray, cellStateArray) = tidyNDArrays {
            val seqLength = input.shape[0]
            val batchSize = input.shape[1]

            val forwardH = activations[2]
            val reverseH = activations[5]

            val forwardLSTMGates = LSTMGates.create(
                weights.unstack()[0],
                recurrentWeights.unstack()[0],
                bias?.unstack()?.get(0),
                peepholes?.unstack()?.get(0),
                batchSize, hiddenSize
            )

            val reverseLSTMGates = LSTMGates.create(
                weights.unstack()[1],
                recurrentWeights.unstack()[1],
                bias?.unstack()?.get(1),
                peepholes?.unstack()?.get(1),
                batchSize, hiddenSize
            )

            val initHiddenState = initialHiddenState ?: NDArrayTFJS.floatZeros(arrayOf(2, batchSize, hiddenSize))
            val initHiddenStateAsLSTMInput = initHiddenState.unstack()

            val lstmStates = LSTMStates(
                LSTMCellState(initialCellState, numDirections = 2, batchSize, hiddenSize),
                LSTMHiddenState(initHiddenState, initHiddenStateAsLSTMInput, listOf(forwardH, reverseH))
            )

            val forwardOutput = forwardLayer.apply(input, lstmStates, forwardLSTMGates, sequenceLens, numDirection = 0, seqLength, batchSize)
            val reverseOutput = reverseLayer.apply(input, lstmStates, reverseLSTMGates, sequenceLens, numDirection = 1, seqLength, batchSize)

            val outputArray = listOf(forwardOutput, reverseOutput).concat(axis = 1)

            arrayOf(
                outputArray,
                lstmStates.hiddenState.data.map { it.stack() }.stack(),
                lstmStates.cellState.data.map { it.stack() }.stack()
            )
        }

        return LSTMLayerOutput(outputArray, hiddenStateArray, cellStateArray)
    }
}
