package io.kinference.tfjs.operators.layer.recurrent.gru

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.tfjs.operators.layer.recurrent.LayerDirection

class BiGRULayer(hiddenSize: Int, activations: List<String>): GRULayerBase(hiddenSize, activations, LayerDirection.BIDIRECTIONAL) {
    init {
        require(activations.size == 4) { "Required number of activations is 4, but ${activations.size} found" }
    }

    private val forwardLayer = GRULayer(hiddenSize, activations.subList(0, 2), LayerDirection.FORWARD)
    private val reverseLayer = GRULayer(hiddenSize, activations.subList(2, 4), LayerDirection.REVERSE)

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
        val gruHiddenState = GRUHiddenState(initialHiddenState, numDirection = 2, batchSize, hiddenSize)

        val (output, lastHiddenState) = tidyNDArrays {
            val (fWeights, rWeights) = weights.unstack()
            val (fRecWeights, rRecWeights) = recurrentWeights.unstack()
            val (fBias, rBias) = bias?.unstack() ?: arrayOfNulls<NumberNDArrayTFJS?>(2)
            val forwardGRUGates = GRUGates.create(
                fWeights,
                fRecWeights,
                fBias,
                batchSize, hiddenSize, linearBeforeReset
            )

            val reverseGRUGates = GRUGates.create(
                rWeights,
                rRecWeights,
                rBias,
                batchSize, hiddenSize, linearBeforeReset
            )

            val forwardOutput = forwardLayer.apply(input, gruHiddenState, forwardGRUGates, sequenceLength, 0, seqLength, batchSize)
            val reverseOutput = reverseLayer.apply(input, gruHiddenState, reverseGRUGates, sequenceLength, 1, seqLength, batchSize)

            forwardGRUGates.close(); reverseGRUGates.close()

            val output = forwardOutput.concat(listOf(reverseOutput), axis = 0) as NumberNDArrayTFJS
            val lastHiddenState = gruHiddenState.data.map { it.stack() }.stack()

            arrayOf(output, lastHiddenState)
        }

        gruHiddenState.close()

        return output to lastHiddenState
    }
}
