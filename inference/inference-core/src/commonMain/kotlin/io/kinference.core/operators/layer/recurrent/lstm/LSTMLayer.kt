package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.runBlocking
import io.kinference.core.operators.activations.Activation
import io.kinference.primitives.types.DataType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class LSTMLayer(hiddenSize: Int, activations: List<String>, direction: String) : LSTMLayerBase(hiddenSize, activations, direction) {
    init {
        require(activations.size == 3)
    }

    override fun apply(
        input: NumberNDArray,
        weights: NumberNDArray,
        recurrentWeights: NumberNDArray,
        bias: NumberNDArray?,
        sequenceLens: IntNDArray?,
        initialHiddenState: NumberNDArray?,
        initialCellState: NumberNDArray?,
        peepholes: NumberNDArray?,
        dataType: DataType,
    ): Triple<NumberNDArray, NumberNDArray, NumberNDArray> {
        val h = Activation.create(activations[2], dataType)

        val seqLength = input.shape[0]
        val batchSize = input.shape[1]
        val outputArray = allocateNDArray(dataType, intArrayOf(seqLength, 1, batchSize, hiddenSize)) as MutableNumberNDArray

        val lstmStates = LSTMStates(
            LSTMCellState(initialCellState, dataType, 1, batchSize, hiddenSize),
            LSTMHiddenState(initialHiddenState, dataType, 1, batchSize, hiddenSize, listOf(h))
        )

        val lstmGates = LSTMGates.create(
            weights.view(0),
            recurrentWeights.view(0),
            bias?.view(0),
            peepholes?.view(0),
            batchSize, hiddenSize, dataType
        )

        apply(input, outputArray, lstmStates, lstmGates, sequenceLens, 0, seqLength, batchSize, dataType)

        return Triple(outputArray, lstmStates.hiddenState.data, lstmStates.cellState.data)
    }

    fun apply(
        input: NumberNDArray,
        output: MutableNumberNDArray,
        lstmStates: LSTMStates,
        lstmGates: LSTMGates,
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
                        lstmGates.input.compute(localInput, lstmStates, f, numDirection, batchNum)
                        lstmGates.forget.compute(localInput, lstmStates, f, numDirection, batchNum)
                        lstmGates.cell.compute(localInput, lstmStates, g, numDirection, batchNum)
                        lstmStates.cellState.compute(lstmGates, numDirection, batchNum)
                        lstmGates.output.compute(localInput, lstmStates, f, numDirection, batchNum)
                        lstmStates.hiddenState.compute(lstmGates, lstmStates.cellState, numDirection, batchNum)
                        val outputVector = lstmStates.hiddenState.getVector(numDirection, batchNum)

                        output.viewMutable(seqNum, numDirection, batchNum).copyFrom(0, outputVector)
                    }
                }
        }

        //TODO: research optimal batchSize for run with coroutines
        for (seqNum in seqRange) {
            if (batchSize > 1) {
                runBlocking(Dispatchers.Default) { wrapper(seqNum) { launch { it() } }  }
            } else {
                wrapper(seqNum)
            }
        }
    }
}
