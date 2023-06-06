package io.kinference.tfjs.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.unstack
import io.kinference.ndarray.unstackAs3DTypedArray
import io.kinference.ndarray.update
import io.kinference.tfjs.operators.activations.activate

private fun init3DZeroState(numDirections: Int, batchSize: Int, hiddenSize: Int): Array<Array<MutableNumberNDArrayTFJS>> {
    return Array(numDirections) { Array(batchSize) { NDArrayTFJS.floatZeros(arrayOf(hiddenSize)).asMutable() } }
}

class LSTMGate internal constructor(
    private val weights: NumberNDArrayTFJS,
    private val recurrentWeights: NumberNDArrayTFJS,
    private val bias: NumberNDArrayTFJS?,
    private val peephole: NumberNDArrayTFJS?,
    batchSize: Int, hiddenSize: Int
) {
    private val gateData = Array(batchSize) { NDArrayTFJS.floatZeros(arrayOf(hiddenSize)).asMutable() }

    suspend fun compute(
        input: NumberNDArrayTFJS,
        lstmStates: LSTMStates,
        activationFunction: String,
        numDirection: Int,
        batchNum: Int
    ) {
        val newGateData = input.dot(weights)
        newGateData.plusAssign(lstmStates.hiddenState.getVector(numDirection, batchNum).dot(recurrentWeights))
        if (bias != null) newGateData.plusAssign(bias)
        if (peephole != null) newGateData.plusAssign(peephole.times(lstmStates.cellState.getVector(numDirection, batchNum)))
        gateData.update(batchNum, newGateData.activate(activationFunction).asMutable())
    }

    fun getVector(batchNum: Int) = gateData[batchNum]
}

data class LSTMGates(val input: LSTMGate, val output: LSTMGate, val forget: LSTMGate, val cell: LSTMGate) {
    companion object {
        suspend fun create(weights: NumberNDArrayTFJS, recurrentWeights: NumberNDArrayTFJS, bias: NumberNDArrayTFJS?,
                           peepholes: NumberNDArrayTFJS?, batchSize: Int, hiddenSize: Int): LSTMGates {

            val weightsArray = weights.unstack()
            val recWeightsArray = recurrentWeights.unstack()
            val biasArray = bias?.unstack() ?: arrayOfNulls<NumberNDArrayTFJS?>(8)
            val peepholesArray = peepholes?.unstack() ?: arrayOfNulls<NumberNDArrayTFJS?>(3)

            val inputGate = LSTMGate(
                weightsArray[0],
                recWeightsArray[0],
                biasArray[0]?.plus(biasArray[4]!!),
                peepholesArray[0],
                batchSize, hiddenSize
            )
            val outputGate = LSTMGate(
                weightsArray[1],
                recWeightsArray[1],
                biasArray[1]?.plus(biasArray[5]!!),
                peepholesArray[1],
                batchSize, hiddenSize
            )
            val forgetGate = LSTMGate(
                weightsArray[2],
                recWeightsArray[2],
                biasArray[2]?.plus(biasArray[6]!!),
                peepholesArray[2],
                batchSize, hiddenSize
            )
            val cellGate = LSTMGate(
                weightsArray[3],
                recWeightsArray[3],
                biasArray[3]?.plus(biasArray[7]!!),
                null,
                batchSize, hiddenSize
            )

            return LSTMGates(inputGate, outputGate, forgetGate, cellGate)
        }
    }
}

class LSTMCellState internal constructor(
    initCellState: NumberNDArrayTFJS?, numDirections: Int,
    batchSize: Int, hiddenSize: Int
) {
    private val stateData = initCellState?.unstackAs3DTypedArray() ?: init3DZeroState(numDirections, batchSize, hiddenSize)

    val data: Array<Array<MutableNumberNDArrayTFJS>>
        get() = stateData

    suspend fun compute(lstmGates: LSTMGates, numDirection: Int, batchNum: Int) {
        val stateLocal = stateData[numDirection][batchNum]

        stateLocal.timesAssign(lstmGates.forget.getVector(batchNum))
        val temp = lstmGates.input.getVector(batchNum).times(lstmGates.cell.getVector(batchNum))
        stateLocal.plusAssign(temp)
        temp.close()
    }

    fun getVector(numDirection: Int, batchNum: Int) = stateData[numDirection][batchNum]
}

class LSTMHiddenState internal constructor(
    initHiddenState: NumberNDArrayTFJS,
    initHiddenStateAsLSTMInput: Array<NumberNDArrayTFJS>,
    private val activationFunctions: List<String>
) {
    private val stateData = initHiddenState.unstackAs3DTypedArray()
    private val stateDataAsLSTMInput = initHiddenStateAsLSTMInput.unstackAs3DTypedArray()

    val data: Array<Array<MutableNumberNDArrayTFJS>>
        get() = stateData

    suspend fun compute(lstmGates: LSTMGates, cellState: LSTMCellState, numDirection: Int, batchNum: Int) {
        val newState = cellState
            .getVector(numDirection, batchNum)
            .activate(activationFunctions[numDirection]).asMutable()
        newState.timesAssign(lstmGates.output.getVector(batchNum))

        stateData[numDirection].update(batchNum, newState)
    }

    suspend fun update(numDirection: Int) {
        val updateData = stateData[numDirection]
        stateDataAsLSTMInput[numDirection] = Array(updateData.size) { updateData[it].clone() }
    }

    fun getVector(numDirection: Int, batchNum: Int): NumberNDArrayTFJS = stateDataAsLSTMInput[numDirection][batchNum]

    fun getVectorRaw(numDirection: Int, batchNum: Int): NumberNDArrayTFJS = data[numDirection][batchNum]

}

data class LSTMStates(val cellState: LSTMCellState, val hiddenState: LSTMHiddenState)

data class LSTMLayerOutput(
    val output: NumberNDArrayTFJS,
    val hiddenState: NumberNDArrayTFJS,
    val cellState: NumberNDArrayTFJS
)
