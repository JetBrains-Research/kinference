package io.kinference.core.operators.activations

import io.kinference.core.attributes.Attribute
import io.kinference.core.operators.*
import io.kinference.ndarray.arrays.NDArray
import io.kinference.ndarray.extensions.allocateNDArray
import kotlin.time.ExperimentalTime

sealed class Identity(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 14)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in IdentityVer1.VERSION.asRange() -> IdentityVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Identity operator: $version")
        }
    }
}

@ExperimentalTime
class IdentityVer1(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Identity(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 14)
        private val INFO = OperatorInfo("Identity", emptyMap(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun activate(input: NDArray): NDArray = allocateNDArray(input.type, input.shape).apply { copyFrom(0, input) }
}
