package io.kinference.core.operators.activations

import io.kinference.core.attributes.Attribute
import io.kinference.core.operators.*
import io.kinference.ndarray.arrays.MutableNumberNDArray
import io.kinference.ndarray.arrays.NDArray
import kotlin.time.ExperimentalTime

sealed class Erf(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(info, attributes, inputs, outputs) {
    companion object {
        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in ErfVer9.VERSION.asRange() -> ErfVer9(attributes, inputs, outputs)
            else -> error("Unsupported version of Erf operator: $version")
        }
    }
}

@ExperimentalTime
class ErfVer9(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Erf(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("Erf", emptySet(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun activate(input: NDArray): NDArray = (input.toMutable() as MutableNumberNDArray).erf()
}
