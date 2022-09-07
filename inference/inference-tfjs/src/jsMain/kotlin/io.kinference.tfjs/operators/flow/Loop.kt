package io.kinference.tfjs.operators.flow

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.*
import io.kinference.tfjs.graph.TFJSGraph

sealed class Loop(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in LoopVer1.VERSION.asRange() -> LoopVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Loop operator: $version")
            }
    }
}

class LoopVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Loop(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("body", setOf(AttributeProto.AttributeType.GRAPH), required = true)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.INT64), "M", optional = true, scalar = true),
            IOInfo(1, setOf(TensorProto.DataType.BOOL), "cond", optional = true, scalar = true),
            VariadicIOInfo(2, TYPE_CONSTRAINTS, "v_initial")
        )

        private val OUTPUTS_INFO = listOf(VariadicIOInfo(0, TYPE_CONSTRAINTS, "v_final_and_scan_outputs", minimumArity = 1))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("Loop", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val body: TFJSGraph by attribute()

    private fun inner(
        contexts: Contexts<TFJSData<*>>,
        body: TFJSGraph,
        counter: Int,
        condition: Boolean,
        modified: MutableList<TFJSTensor>,
        scans: List<MutableList<TFJSTensor>>
    ): Boolean {
        val inputs = ArrayList<TFJSData<*>>().apply {
            add(scalar(counter, "int32").asTensor(body.inputs[0].name))
            add(scalar(condition).asTensor(body.inputs[1].name))
            body.inputs.drop(2).zip(modified) { info, data ->
                add(data.rename(info.name))
            }
        }

        val outputs = body.execute(inputs, contexts) as List<TFJSTensor>

        val newCondition = outputs.first().data.dataBool().first()
        val modifiedInputs = outputs.slice(1 until 1 + body.inputs.size - 2)
        val newScans = outputs.drop(1 + body.inputs.size - 2)

        modified.clear()
        modified.addAll(modifiedInputs)

        require(newScans.size == scans.size) { "Loop subgraph didn't provide expected output count" }
        scans.zip(newScans) { scan, output ->
            scan.add(output)
        }

        return newCondition
    }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidyNDArrays {
            val maxTripCount = inputs[0]?.data?.dataInt()?.first()
            val keepGoing = inputs[1]?.data?.dataBool()?.first()

            require(body.inputs.size == inputs.size) { "Not enough inputs for Loop subgraph\nPresent: ${inputs.size}, Expected: ${body.inputs.size}" }

            val modified = inputs.drop(2).requireNoNulls().map {
                (it.data.clone() as NDArrayTFJS).asTensor(it.name)
            }.toMutableList()

            val modifiedCount = modified.size

            val scansCount = body.outputs.size - 1 - modifiedCount
            val scans = List(scansCount) { ArrayList<TFJSTensor>() }

            var counter = 0
            var condition = keepGoing ?: true

            contexts as Contexts<TFJSData<*>>
            when {
                maxTripCount == null && keepGoing == null -> {
                    while (true) {
                        condition = inner(contexts, body, counter, condition, modified, scans)
                        counter += 1
                    }
                }

                maxTripCount == null && keepGoing != null -> {
                    while (condition) {
                        condition = inner(contexts, body, counter, condition, modified, scans)
                        counter += 1
                    }
                }

                maxTripCount != null && keepGoing == null -> {
                    for (counter in 0 until maxTripCount) {
                        condition = inner(contexts, body, counter, condition, modified, scans)
                    }
                }

                maxTripCount != null && keepGoing != null -> {
                    for (counter in 0 until maxTripCount) {
                        if (!condition) break
                        condition = inner(contexts, body, counter, condition, modified, scans)
                    }
                }
            }

            val stackedScans = scans.map { scan -> scan.map { tensor -> tensor.data }.stack(axis = 0) }

            return@tidyNDArrays (modified.map { it.data } + stackedScans).toTypedArray()
        }

        return outputs.asNamedOutputs(this.outputs)
    }
}
