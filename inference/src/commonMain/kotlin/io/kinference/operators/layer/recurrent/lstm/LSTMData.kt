package io.kinference.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.primitives.types.DataType

abstract class Gate(protected val weights: NumberNDArray,
                    protected val recurrentWeights: NumberNDArray,
                    protected val bias: NumberNDArray?,
                    protected val peephole: NumberNDArray?) {
    abstract fun compute(input: NumberNDArray, LSTMStates: LSTMStates, activationFunction: PrimitiveToPrimitiveFunction, numDirection: Int, batchNum: Int)
    abstract fun getVector(batchNum: Int): NumberNDArray
}

class LSTMGate(weights: NumberNDArray, recurrentWeights: NumberNDArray, bias: NumberNDArray?, peephole: NumberNDArray?, batchSize: Int, hiddenSize: Int, dataType: DataType):
        Gate(weights, recurrentWeights, bias, peephole) {
    private val gateData = allocateNDArray(dataType, intArrayOf(batchSize, hiddenSize)) as MutableNumberNDArray

    override fun compute(input: NumberNDArray, LSTMStates: LSTMStates, activationFunction: PrimitiveToPrimitiveFunction, numDirection: Int, batchNum: Int) {
        val gateLocal = gateData.viewMutable(batchNum)
        gateLocal.clean()

        input.dot(weights, gateLocal)
        LSTMStates.hiddenState.getVector(numDirection, batchNum).dot(recurrentWeights, gateLocal)
        if (bias != null) gateLocal.plusAssign(bias)
        if (peephole != null) gateLocal.plusAssign(peephole.times(LSTMStates.cellState.getVector(numDirection, batchNum)))
        gateLocal.mapMutable(activationFunction)
    }

    override fun getVector(batchNum: Int) = gateData.view(batchNum)
}

data class LSTMGates(val input: Gate, val output: Gate, val forget: Gate, val cell: Gate) {
    companion object {
        fun create(weights: NumberNDArray, recurrentWeights: NumberNDArray, bias: NumberNDArray?, peepholes: NumberNDArray?,
                   batchSize: Int, hiddenSize: Int, dataType: DataType): LSTMGates {
            val inputGate = LSTMGate(
                weights.view(0),
                recurrentWeights.view(0),
                bias?.view(0)?.plus(bias.view(4)),
                peepholes?.view(0),
                batchSize, hiddenSize, dataType
            )
            val outputGate = LSTMGate(
                weights.view(1),
                recurrentWeights.view(1),
                bias?.view(1)?.plus(bias.view(5)),
                peepholes?.view(1), batchSize, hiddenSize, dataType
            )
            val forgetGate = LSTMGate(
                weights.view(2),
                recurrentWeights.view(2),
                bias?.view(2)?.plus(bias.view(6)),
                peepholes?.view(2), batchSize, hiddenSize, dataType
            )
            val cellGate = LSTMGate(
                weights.view(3),
                recurrentWeights.view(3),
                bias?.view(3)?.plus(bias.view(7)),
                null,
                batchSize, hiddenSize, dataType
            )

            return LSTMGates(inputGate, outputGate, forgetGate, cellGate)
        }
    }
}

abstract class State {
    abstract val data: NumberNDArray

    abstract fun compute(LSTMGates: LSTMGates, LSTMStates: LSTMStates, numDirection: Int, batchNum: Int)

    abstract fun getVector(numDirection: Int, batchNum: Int): NumberNDArray
}

class LSTMCellState(initCellState: NumberNDArray?, dataType: DataType, numDirections: Int, batchSize: Int, hiddenSize: Int): State() {
    private val stateData = initCellState?.toMutable() ?: allocateNDArray(dataType, intArrayOf(numDirections, batchSize, hiddenSize)) as MutableNumberNDArray
    private val tempData = allocateNDArray(dataType, intArrayOf(numDirections, batchSize, hiddenSize)) as MutableNumberNDArray

    override val data: NumberNDArray
        get() = stateData

    override fun compute(LSTMGates: LSTMGates, LSTMStates: LSTMStates, numDirection: Int, batchNum: Int) {
        val stateLocal = stateData.viewMutable(numDirection, batchNum)
        val tempLocal = tempData.viewMutable(numDirection, batchNum)

        stateLocal.timesAssign(LSTMGates.forget.getVector(batchNum))
        LSTMGates.input.getVector(batchNum).times(LSTMGates.cell.getVector(batchNum), tempLocal)
        stateLocal.plusAssign(tempLocal)
    }

    override fun getVector(numDirection: Int, batchNum: Int) = stateData.view(numDirection, batchNum)
}

class LSTMHiddenState(initHiddenState: NumberNDArray?, dataType: DataType, numDirection: Int, batchSize: Int, hiddenSize: Int,
                      private val activationFunctions: List<PrimitiveToPrimitiveFunction>): State() {
    private val stateData = initHiddenState?.toMutable() ?: allocateNDArray(dataType, intArrayOf(numDirection, batchSize, hiddenSize)) as MutableNumberNDArray
    override val data: NumberNDArray
        get() = stateData

    override fun compute(LSTMGates: LSTMGates, LSTMStates: LSTMStates, numDirection: Int, batchNum: Int) {
        val stateLocal = stateData.viewMutable(numDirection, batchNum)
        stateLocal.copyFrom(0, LSTMStates.cellState.getVector(numDirection, batchNum))
        stateLocal.mapMutable(activationFunctions[numDirection])
        stateLocal.timesAssign(LSTMGates.output.getVector(batchNum))
    }

    override fun getVector(numDirection: Int, batchNum: Int): NumberNDArray = stateData.view(numDirection, batchNum)

}

data class LSTMStates(val cellState: State, val hiddenState: State)
