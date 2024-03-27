package io.kinference.core.operators.layer.recurrent.gru

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.primitives.types.DataType

class GRUDefaultGate private constructor(
    private val weights: NumberNDArrayCore,
    private val recurrentWeights: NumberNDArrayCore,
    private val gateData: MutableNumberNDArrayCore,
    private val bias: NumberNDArrayCore?,
) {
    suspend fun compute(
        input: NumberNDArrayCore,
        hiddenState: GRUHiddenState,
        activationFunction: PrimitiveToPrimitiveFunction,
        numDirection: Int,
        batchNum: Int
    ) {
        val gateLocal = gateData.viewMutable(batchNum)
        gateLocal.clean()

        input.dot(weights, gateLocal)
        hiddenState.getVector(numDirection, batchNum).dot(recurrentWeights, gateLocal)
        if (bias != null) gateLocal.plusAssign(bias)
        gateLocal.mapMutable(activationFunction)
    }

    fun getVector(batchNum: Int) = gateData.view(batchNum)

    companion object {
        internal suspend operator fun invoke(
            weights: NumberNDArrayCore,
            recurrentWeights: NumberNDArrayCore,
            bias: NumberNDArrayCore?,
            batchSize: Int,
            hiddenSize: Int,
            dataType: DataType
        ): GRUDefaultGate {

            val gateData = allocateNDArray(dataType, intArrayOf(batchSize, hiddenSize)) as MutableNumberNDArrayCore
            val gate = GRUDefaultGate(weights, recurrentWeights, gateData, bias)

            return gate
        }
    }
}

class GRUHiddenGate private constructor(
    private val weights: NumberNDArrayCore,
    private val recurrentWeights: NumberNDArrayCore,
    private val bias: NumberNDArray?,
    private val weightsBias: NumberNDArray?,
    private val recurrentBias: NumberNDArray?,
    private val linearBeforeReset: Boolean,
    private val gateData: MutableNumberNDArrayCore,
    private val tempData: MutableNumberNDArrayCore
) {
    companion object {
        suspend operator fun invoke(
            weights: NumberNDArrayCore,
            recurrentWeights: NumberNDArrayCore,
            wBias: NumberNDArrayCore?,
            rBias: NumberNDArrayCore?,
            batchSize: Int,
            hiddenSize: Int,
            dataType: DataType,
            linearBeforeReset: Boolean
        ): GRUHiddenGate {
            val bias: NumberNDArrayCore?
            val weightsBias: NumberNDArrayCore?
            val recurrentBias: NumberNDArrayCore?

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

            val gateData = allocateNDArray(dataType, intArrayOf(batchSize, hiddenSize)) as MutableNumberNDArrayCore
            val tempData = allocateNDArray(dataType, intArrayOf(batchSize, hiddenSize)) as MutableNumberNDArrayCore

            return GRUHiddenGate(
                weights, recurrentWeights, bias, weightsBias, recurrentBias, linearBeforeReset, gateData, tempData
            )
        }
    }

    suspend fun compute(
        input: NumberNDArrayCore,
        hiddenState: GRUHiddenState,
        gates: GRUGates,
        activationFunction: PrimitiveToPrimitiveFunction,
        numDirection: Int,
        batchNum: Int
    ) =
        if (linearBeforeReset)
            computeWithReset(input, hiddenState, gates, activationFunction, numDirection, batchNum)
        else
            computeDefault(input, hiddenState, gates, activationFunction, numDirection, batchNum)

    private suspend fun computeDefault(
        input: NumberNDArrayCore,
        hiddenState: GRUHiddenState,
        gates: GRUGates,
        activationFunction: PrimitiveToPrimitiveFunction,
        numDirection: Int,
        batchNum: Int
    ) {
        val gateLocal = gateData.viewMutable(batchNum)
        val tempLocal = tempData.viewMutable(batchNum)
        gateLocal.clean()

        input.dot(weights, gateLocal)
        gates.reset.getVector(batchNum).times(hiddenState.getVector(numDirection, batchNum), tempLocal)
        tempLocal.dot(recurrentWeights, gateLocal)
        if (bias != null) gateLocal.plusAssign(bias)
        gateLocal.mapMutable(activationFunction)
    }

    private suspend fun computeWithReset(
        input: NumberNDArrayCore,
        hiddenState: GRUHiddenState,
        gates: GRUGates,
        activationFunction: PrimitiveToPrimitiveFunction,
        numDirection: Int,
        batchNum: Int
    ) {
        val gateLocal = gateData.viewMutable(batchNum)
        gateLocal.clean()

        hiddenState.getVector(numDirection, batchNum).dot(recurrentWeights, gateLocal)
        if (recurrentBias != null) gateLocal.plusAssign(recurrentBias)
        gateLocal.timesAssign(gates.reset.getVector(batchNum))
        input.dot(weights, gateLocal)
        if (weightsBias != null) gateLocal.plusAssign(weightsBias)
        gateLocal.mapMutable(activationFunction)
    }

    fun getVector(batchNum: Int) = gateData.view(batchNum)
}

data class GRUGates(val update: GRUDefaultGate, val reset: GRUDefaultGate, val hidden: GRUHiddenGate) {
    companion object {
        suspend fun create(
            weights: NumberNDArrayCore, recurrentWeights: NumberNDArrayCore, bias: NumberNDArrayCore?,
            batchSize: Int, hiddenSize: Int, dataType: DataType, linearBeforeReset: Boolean
        ): GRUGates {
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

class GRUHiddenState private constructor(
    private val stateData: MutableNumberNDArrayCore,
    private val tempData: MutableNumberNDArrayCore,
    private val dataType: DataType
) {

    val data: NumberNDArrayCore
        get() = stateData

    suspend fun compute(gates: GRUGates, numDirection: Int, batchNum: Int) {
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

        internal suspend operator fun invoke(initHiddenState: NumberNDArrayCore?,
                                             dataType: DataType,
                                             numDirection: Int, batchSize: Int, hiddenSize: Int): GRUHiddenState {

            val stateData = initHiddenState?.toMutable() ?: allocateNDArray(dataType, intArrayOf(numDirection, batchSize, hiddenSize)) as MutableNumberNDArrayCore
            val tempData = allocateNDArray(dataType, intArrayOf(numDirection, batchSize, hiddenSize)) as MutableNumberNDArrayCore

            val state = GRUHiddenState(stateData, tempData, dataType)

            return state
        }

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

data class GRULayerOutput(
    val output: NumberNDArrayCore,
    val hiddenState: NumberNDArrayCore
)
