package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.extensions.*

sealed class Slice(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<TFJSTensor, TFJSTensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in SliceVer10.VERSION.asRange() -> SliceVer10(attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

class SliceVer10(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Slice(INFO, attributes, inputs, outputs) {
    companion object {
        private val DATA_TYPE_CONSTRAINTS = ALL_DATA_TYPES
        private val INDEX_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.INT64, TensorProto.DataType.INT32)

        private val INPUTS_INFO = listOf(
            IOInfo(0, DATA_TYPE_CONSTRAINTS, "data", optional = false, differentiable = true),
            IOInfo(1, INDEX_TYPE_CONSTRAINTS, "starts", optional = false, differentiable = false),
            IOInfo(2, INDEX_TYPE_CONSTRAINTS, "ends", optional = false, differentiable = false),
            IOInfo(3, INDEX_TYPE_CONSTRAINTS, "axes", optional = true, differentiable = false),
            IOInfo(4, INDEX_TYPE_CONSTRAINTS, "steps", optional = true, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, DATA_TYPE_CONSTRAINTS, "output", optional = false, differentiable = true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("Slice", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }


    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data
            val axes = inputs.getOrNull(3)?.data?.dataInt()?.copyOf()?.apply {
                for ((idx, axis) in this.withIndex()) {
                    set(idx, input.indexAxis(axis))
                }
            } ?: IntArray(input.shape.size) { it }

            val incompleteStarts = inputs[1]!!.data.dataInt()
            require(incompleteStarts.size == axes.size)

            val incompleteEnds = inputs[2]!!.data.dataInt()
            require(incompleteEnds.size == axes.size)

            val incompleteSteps = inputs.getOrNull(4)?.data?.dataInt() ?: IntArray(axes.size) { 1 }
            require(incompleteSteps.size == axes.size)

            val starts = IntArray(input.shape.size)
            val ends = IntArray(input.shape.size)
            val steps = IntArray(input.shape.size)

            for (axis in input.shape.indices) {
                val index = axes.indexOf(axis)
                if (index == -1) {
                    starts[axis] = 0
                    ends[axis] = input.shape[axis]
                    steps[axis] = 1
                } else {
                    val step = incompleteSteps[index]
                    steps[axis] = step
                    val start = incompleteStarts[index]
                    val end = incompleteEnds[index]
                    val dim = input.shape[axis]

                    if (step > 0) {
                        if (start >= dim) {
                            starts[axis] = 0
                            ends[axis] = 0
                            continue
                        }

                        val actualStart = if (start < 0) {
                            start + dim
                        } else start

                        val actualEnd = if (end < 0) {
                            end + dim
                        } else end
                        starts[axis] = actualStart
                        ends[axis] = actualEnd
                    } else {
                        starts[axis] = start
                        ends[axis] = end
                    }
                }
            }
            val output = input.slice(starts.toTypedArray(), ends.toTypedArray(), steps.toTypedArray())
            return@tidy arrayOf(output)
        }


        return listOf(outputs[0].asTensor("output"))
    }
}
