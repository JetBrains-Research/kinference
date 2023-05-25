package io.kinference.tfjs.operators.layer.recurrent.lstm

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.graph.GraphContext
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.tidyNDArrays
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.operators.layer.recurrent.LayerDirection

sealed class LSTM(
    name: String, 
    info: OperatorInfo, 
    attributes: Map<String, Attribute<Any>>, 
    inputs: List<String>, 
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 7)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): LSTM {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in LSTMVer7.VERSION.asRange() -> LSTMVer7(name, attributes, inputs, outputs)
                else -> error("Unsupported version of LSTM operator: $version")
            }
        }
    }
}

class LSTMVer7(
    name: String, 
    attributes: Map<String, Attribute<Any>>, 
    inputs: List<String>, 
    outputs: List<String>
) : LSTM(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("activation_alpha", setOf(AttributeProto.AttributeType.FLOATS), required = false, emptyList<Float>()),
            AttributeInfo("activation_beta", setOf(AttributeProto.AttributeType.FLOATS), required = false, emptyList<Float>()),
            AttributeInfo("activations", setOf(AttributeProto.AttributeType.STRINGS),
                required = false, listOf("Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh")
            ),
            AttributeInfo("clip", setOf(AttributeProto.AttributeType.FLOAT), false, Float.MAX_VALUE),
            AttributeInfo("direction", setOf(AttributeProto.AttributeType.STRING), false, "forward"),
            AttributeInfo("hidden_size", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("input_forget", setOf(AttributeProto.AttributeType.INT), false, 0),
            AttributeInfo("layout", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false), // [seq_length, batch_size, input_size]
            IOInfo(1, TYPE_CONSTRAINTS, "W", optional = false), // [num_directions, 4*hidden_size, input_size]
            IOInfo(2, TYPE_CONSTRAINTS, "R", optional = false), // [num_directions, 4*hidden_size, hidden_size]
            IOInfo(3, TYPE_CONSTRAINTS, "B", optional = true), // [num_directions, 8*hidden_size]

            IOInfo(4, setOf(TensorProto.DataType.INT32), "sequence_lens", optional = true), // [batch_size]
            IOInfo(5, TYPE_CONSTRAINTS, "initial_h", optional = true), // [num_directions, batch_size, hidden_size]
            IOInfo(6, TYPE_CONSTRAINTS, "initial_c", optional = true), // [num_directions, batch_size, hidden_size]
            IOInfo(7, TYPE_CONSTRAINTS, "P", optional = true) // [num_directions, 3*hidden_size]
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = true), // [seq_length, num_directions, batch_size, hidden_size]
            IOInfo(1, TYPE_CONSTRAINTS, "Y_h", optional = true), // [num_directions, batch_size, hidden_size]
            IOInfo(2, TYPE_CONSTRAINTS, "Y_c", optional = true) // [num_directions, batch_size, hidden_size]
        )

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("LSTM", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)


        private suspend fun prepareBias(tensor: TFJSTensor): TFJSTensor {
            val shape = tensor.data.shape
            val newShape = intArrayOf(shape[0], 8, shape[1] / 8)
            return tensor.data.reshape(newShape).asTensor("prepared_${tensor.name}")
        }

        private suspend fun preparePeepholes(tensor: TFJSTensor): TFJSTensor {
            val shape = tensor.data.shape
            val newShape = intArrayOf(shape[0], 3, shape[1] / 3)
            return tensor.data.reshape(newShape).asTensor("prepared_${tensor.name}")
        }

        private suspend fun prepareWeights(tensor: TFJSTensor): TFJSTensor {
            val shape = tensor.data.shape
            val newShape = intArrayOf(shape[0], 4, shape[1] / 4, shape[2])
            val transposeShape = intArrayOf(0, 1, 3, 2)
            return tensor.data.reshape(newShape).transpose(transposeShape).asTensor("prepared_${tensor.name}")
        }
    }

    private val activations: List<String> by attribute { it: List<String> ->
        when(direction) {
            LayerDirection.FORWARD, LayerDirection.REVERSE -> it.subList(0, 3)
            LayerDirection.BIDIRECTIONAL -> it
        }
    }

    private val direction: LayerDirection by attribute { it: String -> LayerDirection.valueOf(it.uppercase()) }

    private val hiddenSize: Int by attribute("hidden_size") { it: Number -> it.toInt() }
    private val batchWise: Boolean by attribute("layout") { it: Number -> it.toInt() == 1 }

    init {
        if (batchWise) error("BatchWise LSTM is not supported")
    }

    private val lstmLayer = LSTMLayerBase.create(hiddenSize, activations, direction)

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val (output, lastState, lastCellState) = tidyNDArrays {
            val input = inputs[0]!!.data as NumberNDArrayTFJS

            val weights = inputs[1]!!
            val preparedWeights = prepareWeights(weights).data as NumberNDArrayTFJS

            val recurrentWeights = inputs[2]!!
            val preparedRecurrentWeights = prepareWeights(recurrentWeights).data as NumberNDArrayTFJS

            val bias = inputs.getOrNull(3)
            val preparedBias = if (bias != null) prepareBias(bias) else null

            val peepholes = inputs.getOrNull(7)
            val preparedPeepholes = if (peepholes != null) preparePeepholes(peepholes) else null

            val sequenceLens = inputs.getOrNull(4)
            val initialState = inputs.getOrNull(5)
            val initialCellState = inputs.getOrNull(6)

            val (output, lastState, lastCellState) = lstmLayer.apply(
                input,
                preparedWeights,
                preparedRecurrentWeights,
                preparedBias?.data as NumberNDArrayTFJS?,
                sequenceLens?.data as NumberNDArrayTFJS?,
                initialState?.data as NumberNDArrayTFJS?,
                initialCellState?.data as NumberNDArrayTFJS?,
                preparedPeepholes?.data as NumberNDArrayTFJS?,
            )
            arrayOf(output, lastState, lastCellState)
        }
        
        return listOf(output.asTensor("Y"), lastState.asTensor("Y_h"), lastCellState.asTensor("Y_c"))
    }
}
