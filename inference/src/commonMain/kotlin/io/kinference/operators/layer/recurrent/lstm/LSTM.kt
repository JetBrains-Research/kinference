package io.kinference.operators.layer.recurrent.lstm

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.operators.*
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

@ExperimentalTime
class LSTM(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    // TODO: Support activation alpha and beta
    private val activations: List<String> by attribute()
    private val direction: String by attribute()
    private val hiddenSize: Long by attribute("hidden_size")


    private val layer = when (direction) {
        "forward", "reverse" -> LSTMLayer(hiddenSize.toInt(), activations, direction)
        "bidirectional" -> BiLSTMLayer(hiddenSize.toInt(), activations, direction)
        else -> throw UnsupportedOperationException()
    }


    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("activation_alpha", setOf(AttributeProto.AttributeType.FLOATS), false, emptyList<Float>()),
            AttributeInfo("activation_beta", setOf(AttributeProto.AttributeType.FLOATS), false, emptyList<Float>()),
            AttributeInfo("activations", setOf(AttributeProto.AttributeType.STRINGS), false, listOf("Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh")),
            AttributeInfo("clip", setOf(AttributeProto.AttributeType.FLOAT), false, Float.MAX_VALUE),
            AttributeInfo("direction", setOf(AttributeProto.AttributeType.STRING), false, "forward"),
            AttributeInfo("hidden_size", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("input_forget", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "W", optional = false),
            IOInfo(2, TYPE_CONSTRAINTS, "R", optional = false),
            IOInfo(3, TYPE_CONSTRAINTS, "B", optional = true),

            IOInfo(4, setOf(TensorProto.DataType.INT32), "sequence_lens", optional = true),
            IOInfo(5, TYPE_CONSTRAINTS, "initial_h", optional = true),
            IOInfo(6, TYPE_CONSTRAINTS, "initial_c", optional = true),
            IOInfo(7, TYPE_CONSTRAINTS, "P", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = true),
            IOInfo(1, TYPE_CONSTRAINTS, "Y_h", optional = true),
            IOInfo(2, TYPE_CONSTRAINTS, "Y_c", optional = true)
        )

        private val INFO = OperatorInfo("LSTM", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }


    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        return layer.apply(inputs)
    }
}
