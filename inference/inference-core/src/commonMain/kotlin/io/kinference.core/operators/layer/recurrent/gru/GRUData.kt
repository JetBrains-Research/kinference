package io.kinference.core.operators.layer.recurrent.gru

import io.kinference.graph.asCoroutineContext
import io.kinference.model.ExecutionContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType

class GRUDefaultGate(private val weights: NumberNDArray, private val recurrentWeights: NumberNDArray, private val bias: NumberNDArray?, batchSize: Int, hiddenSize: Int, dataType: DataType) {
    private val gateData = allocateNDArray(dataType, intArrayOf(batchSize, hiddenSize)) as MutableNumberNDArray

    fun compute(
        input: NumberNDArray,
        hiddenState: GRUHiddenState,
        activationFunction: PrimitiveToPrimitiveFunction,
        numDirection: Int,
        batchNum: Int,
        executionContext: ExecutionContext? = null
    ) {
        val gateLocal = gateData.viewMutable(batchNum)
        gateLocal.clean()

        input.dot(weights, gateLocal, executionContext.asCoroutineContext())
        hiddenState.getVector(numDirection, batchNum).dot(recurrentWeights, gateLocal, executionContext.asCoroutineContext())
        if (bias != null) gateLocal.plusAssign(bias)
        gateLocal.mapMutable(activationFunction)
    }

    fun getVector(batchNum: Int) = gateData.view(batchNum)
}

class GRUHiddenGate(private val weights: NumberNDArray, private val recurrentWeights: NumberNDArray, wBias: NumberNDArray?, rBias: NumberNDArray?, batchSize: Int, hiddenSize: Int, dataType: DataType, private val linearBeforeReset: Boolean) {

    private val bias: NumberNDArray?
    private val weightsBias: NumberNDArray?
    private val recurrentBias: NumberNDArray?

    init {
        if (linearBeforeReset) {
            bias = null
            weightsBias = wBias
            recurrentBias = rBias
        } else {
            bias = when {
                wBias != null && rBias != null -> wBias.plus(rBias)
                wBias != null -> wBias
                rBias != null -> rBias
                else -> null
            }
            weightsBias = null
            recurrentBias = null
        }
    }

    private val gateData = allocateNDArray(dataType, intArrayOf(batchSize, hiddenSize)) as MutableNumberNDArray
    private val tempData = allocateNDArray(dataType, intArrayOf(batchSize, hiddenSize)) as MutableNumberNDArray

    fun compute(
        input: NumberNDArray,
        hiddenState: GRUHiddenState,
        gates: GRUGates,
        activationFunction: PrimitiveToPrimitiveFunction,
        numDirection: Int,
        batchNum: Int,
        executionContext: ExecutionContext? = null
    ) =
        if (linearBeforeReset)
            computeWithReset(input, hiddenState, gates, activationFunction, numDirection, batchNum, executionContext)
        else
            computeDefault(input, hiddenState, gates, activationFunction, numDirection, batchNum, executionContext)

    private fun computeDefault(
        input: NumberNDArray,
        hiddenState: GRUHiddenState,
        gates: GRUGates,
        activationFunction: PrimitiveToPrimitiveFunction,
        numDirection: Int,
        batchNum: Int,
        executionContext: ExecutionContext? = null
    ) {
        val gateLocal = gateData.viewMutable(batchNum)
        val tempLocal = tempData.viewMutable(batchNum)
        gateLocal.clean()

        input.dot(weights, gateLocal, executionContext.asCoroutineContext())
        gates.reset.getVector(batchNum).times(hiddenState.getVector(numDirection, batchNum), tempLocal)
        tempLocal.dot(recurrentWeights, gateLocal, executionContext.asCoroutineContext())
        if (bias != null) gateLocal.plusAssign(bias)
        gateLocal.mapMutable(activationFunction)
    }

    private fun computeWithReset(
        input: NumberNDArray,
        hiddenState: GRUHiddenState,
        gates: GRUGates,
        activationFunction: PrimitiveToPrimitiveFunction,
        numDirection: Int,
        batchNum: Int,
        executionContext: ExecutionContext? = null
    ) {
        val gateLocal = gateData.viewMutable(batchNum)
        gateLocal.clean()

        hiddenState.getVector(numDirection, batchNum).dot(recurrentWeights, gateLocal, executionContext.asCoroutineContext())
        if (recurrentBias != null) gateLocal.plusAssign(recurrentBias)
        gateLocal.timesAssign(gates.reset.getVector(batchNum))
        input.dot(weights, gateLocal, executionContext.asCoroutineContext())
        if (weightsBias != null) gateLocal.plusAssign(weightsBias)
        gateLocal.mapMutable(activationFunction)
    }

    fun getVector(batchNum: Int) = gateData.view(batchNum)
}

data class GRUGates(val update: GRUDefaultGate, val reset: GRUDefaultGate, val hidden: GRUHiddenGate) {
    companion object {
        fun create(weights: NumberNDArray, recurrentWeights: NumberNDArray, bias: NumberNDArray?, batchSize: Int, hiddenSize: Int, dataType: DataType, linearBeforeReset: Boolean): GRUGates {
            val updateGate = GRUDefaultGate(
                weights.view(0),
                recurrentWeights.view(0),
                bias?.view(0)?.plus(bias.view(3)),
                batchSize, hiddenSize, dataType
            )

            val resetGate = GRUDefaultGate(
                weights.view(1),
                recurrentWeights.view(1),
                bias?.view(1)?.plus(bias.view(4)),
                batchSize, hiddenSize, dataType
            )

            val hiddenGate = GRUHiddenGate(
                weights.view(2),
                recurrentWeights.view(2),
                bias?.view(2),
                bias?.view(5),
                batchSize, hiddenSize, dataType, linearBeforeReset
            )

            return GRUGates(updateGate, resetGate, hiddenGate)
        }
    }

}

class GRUHiddenState(initHiddenState: NumberNDArray?, private val dataType: DataType, numDirection: Int, batchSize: Int, hiddenSize: Int) {
    private val stateData = initHiddenState?.toMutable() ?: allocateNDArray(dataType, intArrayOf(numDirection, batchSize, hiddenSize)) as MutableNumberNDArray
    private val tempData = allocateNDArray(dataType, intArrayOf(numDirection, batchSize, hiddenSize)) as MutableNumberNDArray

    val data: NumberNDArray
        get() = stateData

    fun compute(gates: GRUGates, numDirection: Int, batchNum: Int) {
        val stateLocal = stateData.viewMutable(numDirection, batchNum)
        val tempLocal = tempData.viewMutable(numDirection, batchNum)

        stateLocal.timesAssign(gates.update.getVector(batchNum))

        when (dataType) {
            DataType.DOUBLE -> gates.update.getVector(batchNum).map(DoubleResetMap, tempLocal)
            DataType.FLOAT -> gates.update.getVector(batchNum).map(FloatResetMap, tempLocal)
            else -> error("Unsupported type: $dataType")
        }
        tempLocal.timesAssign(gates.hidden.getVector(batchNum))

        stateLocal.plusAssign(tempLocal)
    }

    companion object {
        private object FloatResetMap : FloatMap {
            override fun apply(value: Float): Float {
                return 1f - value
            }
        }

        private object DoubleResetMap : DoubleMap {
            override fun apply(value: Double): Double {
                return 1.0 - value
            }
        }
    }


    fun getVector(numDirection: Int, batchNum: Int) = stateData.view(numDirection, batchNum)
}
