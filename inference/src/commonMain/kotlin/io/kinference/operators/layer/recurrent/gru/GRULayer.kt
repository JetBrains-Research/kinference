package io.kinference.operators.layer.recurrent.gru

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.runBlocking
import io.kinference.operators.activations.Activation
import io.kinference.primitives.types.DataType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class GRULayer(hiddenSize: Int, activations: List<String>, direction: String): GRULayerBase(hiddenSize, activations, direction) {
    init {
        require(activations.size == 2)
    }

    override fun apply(
        input: NumberNDArray,
        weights: NumberNDArray,
        recurrentWeights: NumberNDArray,
        bias: NumberNDArray?,
        sequenceLens: IntNDArray?,
        initialHiddenState: NumberNDArray?,
        dataType: DataType,
        linearBeforeReset: Boolean
    ): Pair<NumberNDArray, NumberNDArray> {
        val seqLength = input.shape[0]
        val batchSize = input.shape[1]
        val outputArray = allocateNDArray(dataType, intArrayOf(seqLength, 1, batchSize, hiddenSize)) as MutableNumberNDArray

        val gruState = GRUHiddenState(initialHiddenState, dataType, 1, batchSize, hiddenSize)

        val gruGates = GRUGates.create(
            weights.view(0),
            recurrentWeights.view(0),
            bias?.view(0),
            batchSize, hiddenSize, dataType, linearBeforeReset
        )

        apply(input, outputArray, gruState, gruGates, sequenceLens, 0, seqLength, batchSize, dataType)
        return outputArray to gruState.data
    }

    fun apply(
        input: NumberNDArray,
        output: MutableNumberNDArray,
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
        val seqRange = if (direction == "forward") 0 until seqLength else (0 until seqLength).reversed()

        fun wrapper(seqNum: Int, body: (inner: () -> Unit) -> Unit = { it() }) {
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
                runBlocking(Dispatchers.Default) { wrapper(seqNum) { launch { it() } } }
            } else {
                wrapper(seqNum)
            }
        }
    }
}
