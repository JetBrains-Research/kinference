package io.kinference.tfjs.operators.seq

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.graph.TFJSGraph
import io.kinference.types.ValueTypeInfo

sealed class SequenceMap(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSData<*>, TFJSSequence>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 17)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): SequenceMap {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SequenceMapVer17.VERSION.asRange() -> SequenceMapVer17(name, attributes, inputs, outputs)
                else -> error("Unsupported version of SequenceMap operator: $version")
            }
        }
    }
}


class SequenceMapVer17 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : SequenceMap(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "input_sequence", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE),
            VariadicIOInfo(1, ALL_DATA_TYPES, "additional_inputs", onnxDataTypes = setOf(ONNXDataType.ONNX_TENSOR, ONNXDataType.ONNX_SEQUENCE))
        )

        private val OUTPUTS_INFO = listOf(
            VariadicIOInfo(0, ALL_DATA_TYPES, "out_sequence", onnxDataType = ONNXDataType.ONNX_SEQUENCE)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("body", setOf(AttributeProto.AttributeType.GRAPH), required = true),
        )

        internal val VERSION = VersionInfo(sinceVersion = 17)
        private val INFO = OperatorInfo("SequenceMap", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val body: TFJSGraph by attribute()

    private fun TFJSSequence.zipAsGraphInputs(additionalInputs: List<TFJSData<*>>): List<List<TFJSData<*>>> {
        val inputSize = this.length
        require(additionalInputs.filterIsInstance<TFJSSequence>().all { it.length == inputSize }) { "All input sequences must have the same length" }

        val allInputs = listOf(this) + additionalInputs

        return List(inputSize) { i ->
            List(allInputs.size) { j ->
                when (val inputElement = allInputs[j]) {
                    is TFJSSequence -> inputElement.data[i] as TFJSTensor
                    is TFJSTensor -> inputElement
                    else -> error("Unsupported ONNX data type: ${inputElement.type}")
                }.rename(body.availableInputs[j])
            }
        }
    }

    private fun List<List<TFJSData<*>>>.toOutputs(names: List<String>): List<TFJSSequence> {
        this as? List<List<TFJSTensor>> ?: error("SequenceMap operator supports only Tensor-typed subgraph outputs")

        val outSeqLen = this[0].size
        require(this.all { it.size == outSeqLen }) { "Output lists produced by each iteration must have the same size" }

        val iterators = this.map { it.iterator() }
        return List(outSeqLen) { i ->
            val data = iterators.map { it.next() }
            TFJSSequence(
                name = names[i],
                data = data,
                info = ValueTypeInfo.SequenceTypeInfo(
                    elementType = ValueTypeInfo.TensorTypeInfo(type = data[0].info.type)
                )
            )
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSData<*>?>): List<TFJSSequence?> {
        val inputSeq = inputs[0]!! as TFJSSequence
        val additionalInputs = inputs.drop(1).filterNotNull()
        val graphInputs = inputSeq.zipAsGraphInputs(additionalInputs)

        return graphInputs.map { body.execute(it) }.toOutputs(outputs)
    }
}
