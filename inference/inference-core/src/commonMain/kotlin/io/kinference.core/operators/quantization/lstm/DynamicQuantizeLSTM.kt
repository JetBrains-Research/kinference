package io.kinference.core.operators.quantization.lstm

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.operators.*
import io.kinference.core.operators.layer.recurrent.lstm.LSTMContext
import io.kinference.core.operators.layer.recurrent.lstm.LSTMLayerBase
import io.kinference.ndarray.arrays.*
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class DynamicQuantizeLSTM(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in DynamicQuantizeLSTMVer1.VERSION.asRange() -> DynamicQuantizeLSTMVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of DynamicQuantizeLSTM operator: $version")
        }
    }
}

@OptIn(ExperimentalTime::class)
class DynamicQuantizeLSTMVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : DynamicQuantizeLSTM(INFO, attributes, inputs, outputs) {
    companion object {
        private val BYTE_TYPES = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.INT8
        )

        private val FLOAT_TYPE = setOf(TensorProto.DataType.FLOAT)

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("activation_alpha", setOf(AttributeProto.AttributeType.FLOATS), false, emptyList<Float>()),
            AttributeInfo("activation_beta", setOf(AttributeProto.AttributeType.FLOATS), false, emptyList<Float>()),
            AttributeInfo("activations", setOf(AttributeProto.AttributeType.STRINGS), false, listOf("Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh")),
            AttributeInfo("clip", setOf(AttributeProto.AttributeType.FLOAT), false, Float.MAX_VALUE),
            AttributeInfo("direction", setOf(AttributeProto.AttributeType.STRING), false, "forward"),
            AttributeInfo("hidden_size", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("input_forget", setOf(AttributeProto.AttributeType.INT), false, 0),
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, FLOAT_TYPE, "X", optional = false), // [seq_length, batch_size, input_size]
            IOInfo(1, BYTE_TYPES, "W", optional = false), // [num_directions, 4*hidden_size, input_size]
            IOInfo(2, BYTE_TYPES, "R", optional = false), // [num_directions, 4*hidden_size, hidden_size]
            IOInfo(3, FLOAT_TYPE, "B", optional = true), // [num_directions, 8*hidden_size]

            IOInfo(4, setOf(TensorProto.DataType.INT32), "sequence_lens", optional = true), // [batch_size]
            IOInfo(5, FLOAT_TYPE, "initial_h", optional = true), // [num_directions, batch_size, hidden_size]
            IOInfo(6, FLOAT_TYPE, "initial_c", optional = true), // [num_directions, batch_size, hidden_size]
            IOInfo(7, FLOAT_TYPE, "P", optional = true), // [num_directions, 3*hidden_size]
            IOInfo(8, FLOAT_TYPE, "W_scale", optional = false), // [num_directions] or [num_directions, 4*hidden_size]
            IOInfo(9, BYTE_TYPES, "W_zero_point", optional = false), // [num_directions] or [num_directions, 4*hidden_size]
            IOInfo(10, FLOAT_TYPE, "R_scale", optional = false), // [num_directions] or [num_directions, 4*hidden_size]
            IOInfo(11, BYTE_TYPES, "R_zero_point", optional = false) // [num_directions] or [num_directions, 4*hidden_size]
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, FLOAT_TYPE, "Y", optional = true), // [seq_length, num_directions, batch_size, hidden_size]
            IOInfo(1, FLOAT_TYPE, "Y_h", optional = true), // [num_directions, batch_size, hidden_size]
            IOInfo(2, FLOAT_TYPE, "Y_c", optional = true) // [num_directions, batch_size, hidden_size]
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("DynamicQuantizeLSTM", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }

    private val activations: List<String> by attribute() { it: List<String> ->
        if (direction == "forward" || direction == "reverse")
            it.subList(0, 3)
        else it
    }

    private val direction: String by attribute()
    private val hiddenSize: Int by attribute("hidden_size") { it: Number -> it.toInt() }

    private val numDirections = if (direction == "forward" || direction == "reverse") 1 else 2

    private val lstmLayer = LSTMLayerBase.create(hiddenSize, activations, direction)

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs[0]!!.data as FloatNDArray
        val inputAsLSTMInput = QuantizedLSTMInput.create(input)

        val weights = inputs[1]!!
        val preparedWeights = (context.getOrNullValue("prepared_${weights.name}") ?: LSTMContext.prepareWeights(weights)) as KITensor

        val recurrentWeights = inputs[2]!!
        val preparedRecurrentWeights = (context.getOrNullValue("prepared_${recurrentWeights.name}")
            ?: LSTMContext.prepareWeights(recurrentWeights)) as KITensor

        val bias = inputs.getOrNull(3)
        val preparedBias = bias?.let { context.getOrNullValue("prepared_${it.name}") ?: LSTMContext.prepareBias(it) } as KITensor?

        val peepholes = inputs.getOrNull(7)
        val preparedPeepholes = peepholes?.let { context.getOrNullValue("prepared_${it.name}") ?: LSTMContext.preparePeepholes(it) } as KITensor?

        val sequenceLens = inputs.getOrNull(4)
        val initialState = inputs.getOrNull(5)
        val initialCellState = inputs.getOrNull(6)

        val weightsScale = inputs[8]!!
        val weightsZeroPoint = inputs[9]!!

        val recurrentWeightsScale = inputs[10]!!
        val recurrentWeightsZeroPoint = inputs[11]!!

        require(weightsScale.data.rank == 1 && weightsZeroPoint.data.rank == 1 &&
                recurrentWeightsScale.data.rank == 1 && recurrentWeightsZeroPoint.data.rank == 1)

        val preparedWeightsScale = weightsScale.data.toMutable().reshape(intArrayOf(numDirections, 1))
        val preparedWeightsZeroPoint = weightsZeroPoint.data.toMutable().reshape(intArrayOf(numDirections, 1))
        val preparedRecurrentWeightsScale = recurrentWeightsScale.data.toMutable().reshape(intArrayOf(numDirections, 1))
        val preparedRecurrentWeightsZeroPoint = recurrentWeightsZeroPoint.data.toMutable().reshape(intArrayOf(numDirections, 1))


        val weightsAsLSTMWeights = QuantizedLSTMWeights(
            preparedWeights.data as NumberNDArray,
            preparedWeightsScale as FloatNDArray,
            preparedWeightsZeroPoint as NumberNDArray
        )

        val recurrentWeightsAsLSTMWeights = QuantizedLSTMWeights(
            preparedRecurrentWeights.data as NumberNDArray,
            preparedRecurrentWeightsScale as FloatNDArray,
            preparedRecurrentWeightsZeroPoint as NumberNDArray
        )


        val (output, lastState, lastCellState) = lstmLayer.apply(
            inputAsLSTMInput,
            weightsAsLSTMWeights,
            recurrentWeightsAsLSTMWeights,
            preparedBias?.data as NumberNDArray?,
            sequenceLens?.data as IntNDArray?,
            initialState?.data as NumberNDArray?,
            initialCellState?.data as NumberNDArray?,
            preparedPeepholes?.data as NumberNDArray?,
            input.type
        )
        return listOf(output.asTensor("Y"), lastState.asTensor("Y_h"), lastCellState.asTensor("Y_c"))
    }
}

