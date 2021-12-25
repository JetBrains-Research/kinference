package io.kinference.operator

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.graph.Context
import io.kinference.ndarray.extensions.isScalar
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.properties.ReadOnlyProperty
import kotlin.reflect.KProperty
import kotlin.time.ExperimentalTime

class AttributeInfo(val name: String, val types: Set<AttributeProto.AttributeType>, val required: Boolean = false, val default: Any? = null) {
    init {
        require(types.isNotEmpty()) { "Attribute info must have at least one type constraint!" }
    }
}

open class IOInfo(
    val index: Int, val types: Set<TensorProto.DataType>, val name: String,
    val optional: Boolean = false, val onnxDataType: ONNXDataType = ONNXDataType.ONNX_TENSOR,
    val scalar: Boolean = false, val differentiable: Boolean? = null /* null == undefined, TODO */
) {
    init {
        require(types.isNotEmpty()) { "Input info must have at least one type constraint!" }
    }
}

data class VersionInfo(val sinceVersion: Int, val untilVersion: Int = Int.MAX_VALUE) {
    fun asRange() = sinceVersion until untilVersion
}

class VariadicIOInfo(
    startIndex: Int, types: Set<TensorProto.DataType>, name: String,
    val minimumArity: Int = 0, onnxDataType: ONNXDataType = ONNXDataType.ONNX_TENSOR,
    scalar: Boolean = false, differentiable: Boolean? = null,
    val heterogeneous: Boolean = true
) : IOInfo(startIndex, types, name, minimumArity == 0, onnxDataType, scalar, differentiable)

data class OperatorInfo(
    val name: String,
    val attributes: Map<String, AttributeInfo>,
    val inputs: List<IOInfo>,
    val outputs: List<IOInfo>,
    val versionInfo: VersionInfo,
    val domain: String
) {
    constructor(name: String, attributes: Collection<AttributeInfo>, inputs: List<IOInfo>, outputs: List<IOInfo>, versionInfo: VersionInfo, domain: String)
        : this(name, attributes.associateBy { it.name }, inputs, outputs, versionInfo, domain)

    init {
        val variadicInputIndex = inputs.indexOfFirst { it is VariadicIOInfo }
        require(variadicInputIndex == -1 || variadicInputIndex == inputs.size - 1) { "Variadic input must be last" }

        val variadicOutputIndex = outputs.indexOfFirst { it is VariadicIOInfo }
        require(variadicOutputIndex == -1 || variadicOutputIndex == outputs.size - 1) { "Variadic output must be last" }
    }

    companion object {
        const val DEFAULT_DOMAIN = "ai.onnx"
    }
}

@ExperimentalTime
@Suppress("UNCHECKED_CAST")
abstract class Operator<in T : ONNXData<*, *>, out U : ONNXData<*, *>>(
    val info: OperatorInfo,
    val attributes: Map<String, Attribute<Any>> = emptyMap(),
    val inputs: List<String>,
    val outputs: List<String>
) {
    init {
        for (info in info.attributes.values) {
            if (info.required) require(info.name in attributes) { "Required attribute '${info.name}' not specified in ${this.info.name} operator" }

            attributes[info.name]?.let { attribute ->
                require(attribute.type in info.types) { "Attribute '${attribute.name}' type doesn't match specification\nPresent: ${attribute.type}, Expected: one of ${info.types}" }
            }
        }

        for (attribute in attributes.values) {
            if (attribute.name !in info.attributes) {
                println("Unknown attribute '${attribute.name}' in ${info.name} operator")
            }
        }
    }

    private fun check(constraints: List<IOInfo>, values: List<ONNXData<*, *>?>, what: String) {
        fun infos(constraints: List<IOInfo>) = sequence {
            for (constraint in constraints) {
                while (constraint is VariadicIOInfo) yield(constraint)
                yield(constraint)
            }

            yield(null)
        }

        var variadicCounter = 0
        var variadicType: TensorProto.DataType? = null
        infos(constraints).zip(values.asSequence().plusElement(null)) { constraint, value ->
            // TODO check for not null variadic
            if (value == null) {
                require(constraint == null || constraint.optional || (constraint is VariadicIOInfo && variadicCounter >= constraint.minimumArity)) {
                    "Required $what '${constraint!!.name}' for '${info.name}' operator not provided"
                }
                return@zip
            }

            requireNotNull(constraint) { "Unexpected $what '${value.name}' for '${info.name}' operator" }

            if (constraint is VariadicIOInfo) {
                //if (variadicCounter == 0) variadicType = value.info.type
                variadicCounter++

                //if (!constraint.heterogeneous) require(value.info.type == variadicType) { "All ${what}s for '${constraint.name}' must have same type\nPresent: ${value.info.type}, Expected: $variadicType" }
            }

            require(value.type == constraint.onnxDataType) { "Wrong $what ONNX data type '${value.name}' for '${info.name}' operator\nPresent: ${value.type}, Expected: ${constraint.onnxDataType}" }
        }
    }

    fun <D : ONNXData<*, *>> applyWithCheck(context: Context<D>, inputs: List<T?>, profilingContext: ProfilingContext?): List<U?> {
        check(info.inputs, inputs, "input")
        val outputs = apply(context, inputs, profilingContext)
        require(outputs.size >= this.outputs.size) { "Operator '${info.name}' doesn't provide expected output size\nPresent: ${outputs.size}, Expected: at least ${this.outputs.size}" }
        check(info.outputs, outputs, "output")
        return outputs
    }

    suspend fun <D : ONNXData<*, *>> applyWithCheckSuspend(context: Context<D>, inputs: List<T?>, profilingContext: ProfilingContext?): List<U?> {
        check(info.inputs, inputs, "input")
        val outputs = applySuspend(context, inputs, profilingContext)
        require(outputs.size >= this.outputs.size) { "Operator '${info.name}' doesn't provide expected output size\nPresent: ${outputs.size}, Expected: at least ${this.outputs.size}" }
        check(info.outputs, outputs, "output")
        return outputs
    }

    class AttributeValueDelegate<Input, Output>(
        val name: String?, val transform: (Input) -> Output,
        val getValue: Operator<*, *>.(String) -> Any?
    ) : ReadOnlyProperty<Operator<*, *>, Output> {
        private var initialized: Boolean = false
        private var value: Output? = null

        override fun getValue(thisRef: Operator<*, *>, property: KProperty<*>): Output {
            if (!initialized) {
                value = transform(thisRef.getValue(name ?: property.name) as Input)
                initialized = true
            }
            return value as Output
        }
    }

    fun <O> attribute(name: String? = null) = AttributeValueDelegate<O, O>(name, { it }, { getAttribute(it) })
    fun <I, O> attribute(name: String? = null, transform: (I) -> O) = AttributeValueDelegate(name, transform) { getAttribute(it) }

    /**
     * Get attribute from operator info
     *
     * Consider using [attribute] that performs caching on delegate side.
     */
    fun <T> getAttribute(key: String): T {
        val value = getAttributeOrNull<T>(key)
        requireNotNull(value) { "Attribute '$key' not found or don't have a default value" }
        return value
    }

    fun <O> attributeOrNull(name: String? = null) = AttributeValueDelegate<O, O?>(name, { it }, { getAttributeOrNull(it) })
    fun <I, O> attributeOrNull(name: String? = null, transform: (I?) -> O?) = AttributeValueDelegate(name, transform) { getAttributeOrNull(it) }

    /**
     * Get attribute from operator info or null if no default
     *
     * Consider using [attributeOrNull] that performs caching on delegate side
     */
    fun <T> getAttributeOrNull(key: String): T? {
        val info = info.attributes[key]
        requireNotNull(info) { "Attribute '$key' not specified in the '${this.info.name}' operator" }

        return attributes[key]?.value as T? ?: if (!info.required) info.default as T? else null
    }

    abstract fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<T?>, profilingContext: ProfilingContext? = null): List<U?>
    open fun <D : ONNXData<*, *>> apply(context: Context<D>, vararg inputs: T?, profilingContext: ProfilingContext? = null): Collection<U?> =
        apply(context, inputs.toList(), profilingContext)

    open suspend fun <D : ONNXData<*, *>> applySuspend(context: Context<D>, inputs: List<T?>, profilingContext: ProfilingContext? = null): List<U?> =
        apply(context, inputs, profilingContext)
    open suspend fun <D : ONNXData<*, *>> applySuspend(context: Context<D>, vararg inputs: T?, profilingContext: ProfilingContext? = null): Collection<U?> =
        applySuspend(context, inputs.toList(), profilingContext)

    companion object {
        val ALL_DATA_TYPES = TensorProto.DataType.values().toHashSet() - TensorProto.DataType.UNDEFINED
        val PRIMITIVE_DATA_TYPES = setOf(
            TensorProto.DataType.BOOL,
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT16,
            TensorProto.DataType.INT8,
            TensorProto.DataType.INT64,
            TensorProto.DataType.UINT16,
            TensorProto.DataType.UINT8,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64
        )
        val FLOAT_DATA_TYPES = setOf(TensorProto.DataType.BFLOAT16, TensorProto.DataType.FLOAT16, TensorProto.DataType.FLOAT, TensorProto.DataType.DOUBLE)
    }
}
