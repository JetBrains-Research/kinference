package io.kinference.tfjs.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.tfjs.operators.layer.recurrent.LayerDirection

class LSTMLayer(hiddenSize: Int, activations: List<String>, direction: LayerDirection) : LSTMLayerBase(hiddenSize, activations, direction) {
    init {
        require(activations.size == 3) { "Required number of activations is 3, but ${activations.size} found" }
    }

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
            val h = activations[2]

            val seqLength = input.shape[0]
            val batchSize = input.shape[1]

            val stateShape = intArrayOf(1, batchSize, hiddenSize)

            val initHiddenState = initialHiddenState ?: NDArrayTFJS.floatZeros(stateShape.toTypedArray())
            val initHiddenStateAsLSTMInput = arrayOf(initHiddenState.unstack()[0])

            val lstmStates = LSTMStates(
                LSTMCellState(initialCellState, numDirections = 1, batchSize, hiddenSize),
                LSTMHiddenState(initHiddenState, initHiddenStateAsLSTMInput, listOf(h))
            )

            val lstmGates = LSTMGates.create(
                weights.unstack()[0],
                recurrentWeights.unstack()[0],
                bias?.unstack()?.get(0),
                peepholes?.unstack()?.get(0),
                batchSize, hiddenSize
            )

            val outputArray = apply(input, lstmStates, lstmGates, sequenceLens, numDirection = 0, seqLength, batchSize)

            arrayOf(
                outputArray,
                lstmStates.hiddenState.data.flatten().stack().reshape(stateShape),
                lstmStates.cellState.data.flatten().stack().reshape(stateShape)
            )
        }

        return LSTMLayerOutput(outputArray, hiddenStateArray, cellStateArray)
    }

    internal suspend fun apply(
        input: NumberNDArrayTFJS,
        lstmStates: LSTMStates,
        lstmGates: LSTMGates,
        sequenceLens: NumberNDArrayTFJS?,
        numDirection: Int,
        seqLength: Int,
        batchSize: Int,
    ): NumberNDArrayTFJS {
        val (f, g) = activations

        val seqLens = sequenceLens?.dataInt() ?: IntArray(batchSize) { seqLength }
        val seqRange = if (direction == LayerDirection.FORWARD) 0 until seqLength else (0 until seqLength).reversed()

        val inputArray = input.unstack().map { it.unstack() }

        val outputShape = intArrayOf(seqLength, 1, batchSize, hiddenSize)
        val outputs = Array(seqLength) { Array<NumberNDArrayTFJS?>(batchSize) { null } }
        for (seqNum in seqRange) {
            for (batchNum in 0 until batchSize) {
                if (seqNum >= seqLens[batchNum]) continue
                val localInput = inputArray[seqNum][batchNum]
                lstmGates.input.compute(localInput, lstmStates, f, numDirection, batchNum)
                lstmGates.forget.compute(localInput, lstmStates, f, numDirection, batchNum)
                lstmGates.cell.compute(localInput, lstmStates, g, numDirection, batchNum)
                lstmStates.cellState.compute(lstmGates, numDirection, batchNum)
                lstmGates.output.compute(localInput, lstmStates, f, numDirection, batchNum)
                lstmStates.hiddenState.compute(lstmGates, lstmStates.cellState, numDirection, batchNum)
                val outputVector = lstmStates.hiddenState.getVectorRaw(numDirection, batchNum)
                outputs[seqNum][batchNum] = outputVector
            }
            lstmStates.hiddenState.update(numDirection)
        }
        return outputs.map { it.filterNotNull() }.flatten().stack().reshape(outputShape)
    }
}
