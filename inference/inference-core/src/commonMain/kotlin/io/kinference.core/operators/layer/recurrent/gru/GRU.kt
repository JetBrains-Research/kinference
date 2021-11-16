package io.kinference.core.operators.layer.recurrent.gru

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class GRU(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in GRUVer7.VERSION.asRange() -> GRUVer7(attributes, inputs, outputs)
            else -> error("Unsupported version of GRU operator: $version")
        }
    }
}

@OptIn(ExperimentalTime::class)
class GRUVer7(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : GRU(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("activation_alpha", setOf(AttributeProto.AttributeType.FLOATS), false, emptyList<Float>()),
            AttributeInfo("activation_beta", setOf(AttributeProto.AttributeType.FLOATS), false, emptyList<Float>()),
            AttributeInfo("activations", setOf(AttributeProto.AttributeType.STRINGS), false, listOf("Sigmoid", "Tanh", "Sigmoid", "Tanh")),
            AttributeInfo("clip", setOf(AttributeProto.AttributeType.FLOAT), false, Float.MAX_VALUE),
            AttributeInfo("direction", setOf(AttributeProto.AttributeType.STRING), false, "forward"),
            AttributeInfo("hidden_size", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("layout", setOf(AttributeProto.AttributeType.INT), false, 0),
            AttributeInfo("linear_before_reset", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false), // [seq_length, batch_size, input_size]
            IOInfo(1, TYPE_CONSTRAINTS, "W", optional = false), // [num_directions, 4*hidden_size, input_size]
            IOInfo(2, TYPE_CONSTRAINTS, "R", optional = false), // [num_directions, 4*hidden_size, hidden_size]
            IOInfo(3, TYPE_CONSTRAINTS, "B", optional = true), // [num_directions, 8*hidden_size]

            IOInfo(4, setOf(TensorProto.DataType.INT32), "sequence_lens", optional = true), // [batch_size]
            IOInfo(5, TYPE_CONSTRAINTS, "initial_h", optional = true), // [num_directions, batch_size, hidden_size]
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = true), // [seq_length, num_directions, batch_size, hidden_size]
            IOInfo(1, TYPE_CONSTRAINTS, "Y_h", optional = true), // [num_directions, batch_size, hidden_size]
        )

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("GRU", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val activations: List<String> by attribute() { it: List<String> ->
        if (direction == "forward" || direction == "reverse")
            it.subList(0, 2)
        else
            it
    }
    private val direction: String by attribute()
    private val hiddenSize: Int by attribute("hidden_size") { it: Number -> it.toInt() }
    private val batchWise: Boolean by attribute("layout") { it: Number -> it.toInt() == 1 }
    private val linearBeforeReset: Boolean by attribute("linear_before_reset") { it: Number -> it.toInt() == 1 }

    init {
        if (batchWise) error("BatchWise GRU not supported")
    }

    private val gruLayer = GRULayerBase.create(hiddenSize, activations, direction)

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs[0]!!

        val weights = inputs[1]!!
        val preparedWeights = (context.getOrNullValue("prepared_${weights.name}") ?: GRUContext.prepareWeights(weights))

        val recurrentWeights = inputs[2]!!
        val preparedRecurrentWeights = (context.getOrNullValue("prepared_${recurrentWeights.name}")
            ?: GRUContext.prepareWeights(recurrentWeights)) as KITensor

        val bias = inputs.getOrNull(3)
        val preparedBias = bias?.let { context.getOrNullValue("prepared_${it.name}") ?: GRUContext.prepareBias(it) }

        val sequenceLens = inputs.getOrNull(4)
        val initialHiddenState = inputs.getOrNull(5)

        val (output, lastState) = gruLayer.apply(
            input.data as NumberNDArray,
            preparedWeights.data as NumberNDArray,
            preparedRecurrentWeights.data as NumberNDArray,
            preparedBias?.data as NumberNDArray?,
            sequenceLens?.data as IntNDArray?,
            initialHiddenState?.data as NumberNDArray?,
            input.data.type,
            linearBeforeReset
        )

        return listOf(output.asTensor("Y"), lastState.asTensor("Y_h"))
    }
}
