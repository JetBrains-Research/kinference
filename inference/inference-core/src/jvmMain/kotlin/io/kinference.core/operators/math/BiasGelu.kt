package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.MutableNumberNDArrayCore
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.ndarray.arrays.memory.contexts.ManualAllocatorContext
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.gelu.biasGelu
import io.kinference.operator.*
import io.kinference.utils.PredictionContext
import kotlin.coroutines.coroutineContext

sealed class BiasGelu(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in BiasGeluVer1.VERSION.asRange() -> BiasGeluVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of BiasGelu operator: $version")
        }
    }
}


class BiasGeluVer1(name: String, attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : BiasGelu(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("BiasGelu", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.ORT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val manualContext = coroutineContext[PredictionContext.Key] as? ManualAllocatorContext

        val input = inputs[0]!!.data as NumberNDArrayCore
        val bias = inputs[1]!!.data as NumberNDArrayCore

        require(input.shape.last() == bias.shape.last()) { "Last dimensions of input and bias tensors must be equal" }

        val dest = (manualContext?.getNDArray(input.type, input.strides) ?: allocateNDArray(input.type, input.strides)) as MutableNumberNDArrayCore

        // Uses ERF formula with fractional error less than x.xx * 10 ^ -4.
        // Algorithm 26.2.17 in Abromowitz and Stegun, Handbook of Mathematical.
        // Another possible ERF implementation (several ms faster):
        // https://github.com/apache/commons-numbers/blob/master/commons-numbers-gamma/src/main/java/org/apache/commons/numbers/gamma/BoostErf.java

        return listOf(biasGelu(input, bias, dest).asTensor("C", manualContext))
    }
}
