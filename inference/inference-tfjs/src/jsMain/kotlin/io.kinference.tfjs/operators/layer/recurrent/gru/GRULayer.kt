package io.kinference.tfjs.operators.layer.recurrent.gru

import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.*
import io.kinference.tfjs.operators.layer.recurrent.LayerDirection

class GRULayer(hiddenSize: Int, activations: List<String>, direction: LayerDirection): GRULayerBase(hiddenSize, activations, direction) {
    init {
        require(activations.size == 2) { "Required number of activation functions is 2, but ${activations.size} found" }
    }

    override suspend fun apply(
        input: NumberNDArrayTFJS,
        weights: NumberNDArrayTFJS,
        recurrentWeights: NumberNDArrayTFJS,
        bias: NumberNDArrayTFJS?,
        sequenceLength: NumberNDArrayTFJS?,
        initialHiddenState: NumberNDArrayTFJS?,
        linearBeforeReset: Boolean
    ): Pair<NumberNDArrayTFJS, NumberNDArrayTFJS> {
        val seqLength = input.shape[0]
        val batchSize = input.shape[1]

        val (outputArray, lastState) = tidyNDArrays {
            val gruState = GRUHiddenState(initialHiddenState, numDirection = 1, batchSize, hiddenSize)

            val gruGates = GRUGates.create(
                weights.unstack()[0],
                recurrentWeights.unstack()[0],
                bias?.unstack()?.get(0),
                batchSize, hiddenSize, linearBeforeReset
            )

            val outputArray = apply(input, gruState, gruGates, sequenceLength, 0, seqLength, batchSize)
            val lastState = gruState.data.map { it.stack() }.stack()

            gruGates.close()
            gruState.close()

            arrayOf(outputArray, lastState)
        }

        return outputArray to lastState
    }

    internal suspend fun apply(
        input: NumberNDArrayTFJS,
        hiddenState: GRUHiddenState,
        gruGates: GRUGates,
        sequenceLens: NumberNDArrayTFJS?,
        numDirection: Int,
        seqLength: Int,
        batchSize: Int,
    ): NumberNDArrayTFJS {
        val (f, g) = activations

        val seqLens = sequenceLens?.dataInt() ?: IntArray(batchSize) { seqLength }
        val seqRange = if (direction == LayerDirection.FORWARD) 0 until seqLength else (0 until seqLength).reversed()

        val outputShape = intArrayOf(seqLength, 1, batchSize, hiddenSize)
        val outputs = Array(seqLength) { Array<NumberNDArrayTFJS?>(batchSize) { null } }
        for (seqNum in seqRange) {
            for (batchNum in 0 until batchSize) {
                if (seqNum >= seqLens[batchNum]) continue
                val localInput = input.view(seqNum, batchNum)
                gruGates.update.compute(localInput, hiddenState, f, numDirection, batchNum)
                gruGates.reset.compute(localInput, hiddenState, f, numDirection, batchNum)
                gruGates.hidden.compute(localInput, hiddenState, gruGates, g, numDirection, batchNum)
                hiddenState.compute(gruGates, numDirection, batchNum)
                outputs[seqNum][batchNum] = hiddenState.getVector(numDirection, batchNum)
            }
        }

        return outputs.map { it.filterNotNull().stack() }.stack().reshape(outputShape)
    }
}
