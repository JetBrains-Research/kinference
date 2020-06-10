package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import AttributeProto
import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.AttributeInfo
import org.jetbrains.research.kotlin.mpp.inference.operators.InputInfo
import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.operators.OutputInfo
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class LSTM(attributes: Map<String, Attribute<Any>>) : Operator("LSTM", attributes, ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO) {
    val layer: LSTMLayer<Number> = when (attributes["direction"]?.value) {
        "forward" -> LSTMLayer()
        "bidirectional" -> BiLSTMLayer()
        else -> LSTMLayer()
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
            AttributeInfo("activations", setOf(AttributeProto.AttributeType.STRINGS), false, listOf("Sigmoid", "Tanh", "Tanh")),
            AttributeInfo("clip", setOf(AttributeProto.AttributeType.FLOAT), false, Float.MAX_VALUE),
            AttributeInfo("direction", setOf(AttributeProto.AttributeType.STRING), false, "forward"),
            AttributeInfo("hidden_size", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("input_forget", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            InputInfo(0, TYPE_CONSTRAINTS, "X", true),
            InputInfo(1, TYPE_CONSTRAINTS, "W", true),
            InputInfo(2, TYPE_CONSTRAINTS, "R", true),
            InputInfo(3, TYPE_CONSTRAINTS, "B", false, 0),

            InputInfo(4, setOf(TensorProto.DataType.INT32), "sequence_lens", false, emptyList<Int>()),
            InputInfo(5, TYPE_CONSTRAINTS, "initial_h", false, 0),
            InputInfo(6, TYPE_CONSTRAINTS, "initial_c", false, 0),
            InputInfo(7, TYPE_CONSTRAINTS, "P", false, 0)
        )

        private val OUTPUTS_INFO = listOf(
            OutputInfo(0, TYPE_CONSTRAINTS, "Y"),
            OutputInfo(1, TYPE_CONSTRAINTS, "Y_h"),
            OutputInfo(2, TYPE_CONSTRAINTS, "Y_c")
        )
    }

    override fun apply(inputs: Collection<Tensor>): Collection<Tensor> {
        // TODO: use attributes to set up layer
        return layer.apply(inputs)
    }
}
