package io.kinference.core.operators.layer.recurrent.gru

import io.kinference.core.operators.activations.Activation
import io.kinference.core.operators.layer.recurrent.LayerDirection
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.primitives.types.DataType
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

class GRULayer(hiddenSize: Int, activations: List<String>, direction: LayerDirection): GRULayerBase(hiddenSize, activations, direction) {
    init {
        require(activations.size == 2)
    }

    override suspend fun apply(
        input: NumberNDArrayCore,
        weights: NumberNDArrayCore,
        recurrentWeights: NumberNDArrayCore,
        bias: NumberNDArrayCore?,
        sequenceLength: IntNDArray?,
        initialHiddenState: NumberNDArrayCore?,
        dataType: DataType,
        linearBeforeReset: Boolean
    ): GRULayerOutput {
        val seqLength = input.shape[0]
        val batchSize = input.shape[1]
        val outputArray = allocateNDArray(dataType, intArrayOf(seqLength, 1, batchSize, hiddenSize)) as MutableNumberNDArrayCore

        val gruState = GRUHiddenState(initialHiddenState, dataType, 1, batchSize, hiddenSize)

        val gruGates = GRUGates.create(
            weights.view(0),
            recurrentWeights.view(0),
            bias?.view(0),
            batchSize, hiddenSize, dataType, linearBeforeReset
        )

        apply(input, outputArray, gruState, gruGates, sequenceLength, 0, seqLength, batchSize, dataType)
        return GRULayerOutput(outputArray, gruState.data)
    }

    suspend fun apply(
        input: NumberNDArrayCore,
        output: MutableNumberNDArrayCore,
        hiddenState: GRUHiddenState,
        gruGates: GRUGates,
        sequenceLens: IntNDArray?,
        numDirection: Int,
        seqLength: Int,
        batchSize: Int,
        dataType: DataType
    ) {
        val (f, g) = activations.map { Activation.create(it, dataType) }

        val seqLens = sequenceLens?.array?.toArray() ?: IntArray(batchSize) { seqLength }
        val seqRange = if (direction == LayerDirection.FORWARD) 0 until seqLength else (0 until seqLength).reversed()

        suspend fun wrapper(seqNum: Int, body: suspend (inner: suspend () -> Unit) -> Unit = { it() }) {
            for (batchNum in 0 until batchSize) {
                if (seqNum >= seqLens[batchNum]) continue
                body {
                    val localInput = input.view(seqNum, batchNum)
                    gruGates.update.compute(localInput, hiddenState, f, numDirection, batchNum)
                    gruGates.reset.compute(localInput, hiddenState, f, numDirection, batchNum)
                    gruGates.hidden.compute(localInput, hiddenState, gruGates, g, numDirection, batchNum)
                    hiddenState.compute(gruGates, numDirection, batchNum)
                    val outputVector = hiddenState.getVector(numDirection, batchNum)

                    output.viewMutable(seqNum, numDirection, batchNum).copyFrom(0, outputVector)
                }
            }
        }

        //TODO: research optimal batchSize for run with coroutines
        for (seqNum in seqRange) {
            if (batchSize > 1) {
                coroutineScope { wrapper(seqNum) { launch { it() } } }
            } else {
                wrapper(seqNum)
            }
        }
    }
}
