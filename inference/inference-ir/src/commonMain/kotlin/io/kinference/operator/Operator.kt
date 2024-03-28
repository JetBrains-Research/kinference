package io.kinference.operator

import io.kinference.attribute.Attribute
import io.kinference.data.*
import io.kinference.graph.Contexts
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.*
import kotlin.properties.ReadOnlyProperty
import kotlin.reflect.KProperty

class AttributeInfo(val name: String, val types: Set<AttributeProto.AttributeType>, val required: Boolean = false, val default: Any? = null) : Closeable {
    init {
        require(types.isNotEmpty()) { "Attribute info must have at least one type constraint!" }
    }

    override suspend fun close() {
        if (default is Closeable) return default.close()

        if (default is List<*>) {
            for (value in default) {
                if (value is Closeable) value.close()
            }
        }
    }
}

open class IOInfo(
    val index: Int, val types: Set<TensorProto.DataType>, val name: String,
    val optional: Boolean = false, val onnxDataTypes: Set<ONNXDataType> = setOf(ONNXDataType.ONNX_TENSOR),
    val scalar: Boolean = false, val differentiable: Boolean? = null /* null == undefined, TODO */
) {
    init {
        require(types.isNotEmpty()) { "Input info must have at least one type constraint!" }
    }

    companion object {
        operator fun invoke(
            index: Int, types: Set<TensorProto.DataType>, name: String, optional: Boolean = false,
            onnxDataType: ONNXDataType = ONNXDataType.ONNX_TENSOR, scalar: Boolean = false, differentiable: Boolean? = null
        ) = IOInfo(index, types, name, optional, setOf(onnxDataType), scalar, differentiable)
    }
}

data class VersionInfo(val sinceVersion: Int, val untilVersion: Int = Int.MAX_VALUE) {
    fun asRange() = sinceVersion until untilVersion
}

class VariadicIOInfo(
    startIndex: Int, types: Set<TensorProto.DataType>, name: String,
    val minimumArity: Int = 0, onnxDataTypes: Set<ONNXDataType> = setOf(ONNXDataType.ONNX_TENSOR),
    scalar: Boolean = false, differentiable: Boolean? = null, val heterogeneous: Boolean = true
) : IOInfo(startIndex, types, name, minimumArity == 0, onnxDataTypes, scalar, differentiable) {
    companion object {
        operator fun invoke(
            startIndex: Int, types: Set<TensorProto.DataType>, name: String,
            minimumArity: Int = 0, onnxDataType: ONNXDataType = ONNXDataType.ONNX_TENSOR,
            scalar: Boolean = false, differentiable: Boolean? = null, heterogeneous: Boolean = true
        ) = VariadicIOInfo(startIndex, types, name, minimumArity, setOf(onnxDataType), scalar, differentiable, heterogeneous)
    }
}

data class OperatorInfo(
    val type: String,
    val attributes: Map<String, AttributeInfo>,
    val inputs: List<IOInfo>,
    val outputs: List<IOInfo>,
    val versionInfo: VersionInfo,
    val domain: String
) {
    constructor(type: String, attributes: Collection<AttributeInfo>, inputs: List<IOInfo>, outputs: List<IOInfo>, versionInfo: VersionInfo, domain: String)
        : this(type, attributes.associateBy { it.name }, inputs, outputs, versionInfo, domain)

    init {
        val variadicInputIndex = inputs.indexOfFirst { it is VariadicIOInfo }
        require(variadicInputIndex == -1 || variadicInputIndex == inputs.size - 1) { "Variadic input must be last" }

        val variadicOutputIndex = outputs.indexOfFirst { it is VariadicIOInfo }
        require(variadicOutputIndex == -1 || variadicOutputIndex == outputs.size - 1) { "Variadic output must be last" }
    }

    companion object {
        const val DEFAULT_DOMAIN = "ai.onnx"
        const val ML_DOMAIN = "ai.onnx.ml"
        const val ORT_DOMAIN = "com.microsoft"
    }
}


@Suppress("UNCHECKED_CAST")
abstract class Operator<in T : ONNXData<*, *>, out U : ONNXData<*, *>>(
    val name: String,
    val info: OperatorInfo,
    val attributes: Map<String, Attribute<Any>> = emptyMap(),
    inputs: List<String>,
    outputs: List<String>
) : Closeable {
    private val _inputs: ArrayList<String> = ArrayList(inputs)
    private val _outputs: ArrayList<String> = ArrayList(outputs)

    val inputs: List<String>
        get() = _inputs.toList()

    val outputs: List<String>
        get() = _outputs.toList()

    val type: String
        get() = info.type

    init {
        for (info in info.attributes.values) {
            if (info.required) require(info.name in attributes) { "Required attribute '${info.name}' not specified in ${this.info.type} operator" }

            attributes[info.name]?.let { attribute ->
                require(attribute.type in info.types) {
                    "Attribute '${attribute.name}' type doesn't match specification\nPresent: ${attribute.type}, Expected: one of ${info.types}"
                }
            }
        }

        for (attribute in attributes.values) {
            if (attribute.name !in info.attributes) {
                logger.debug { "Unknown attribute '${attribute.name}' in ${info.type} operator" }
            }
        }
    }

    fun renameInput(name: String, newName: String) {
        val idx = _inputs.indexOf(name)
        require(idx != -1) { "Input $name was not found in ${this.name} operator inputs list" }

        _inputs[idx] = newName
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
                    "Required $what '${constraint!!.name}' for '${info.type}' operator not provided"
                }
                return@zip
            }

            requireNotNull(constraint) { "Unexpected $what '${value.name}' for '${info.type}' operator" }

            if (constraint is VariadicIOInfo) {
                //if (variadicCounter == 0) variadicType = value.info.type
                variadicCounter++

                //if (!constraint.heterogeneous) require(value.info.type == variadicType) { "All ${what}s for '${constraint.name}' must have same type\nPresent: ${value.info.type}, Expected: $variadicType" }
            }

            require(value.type in constraint.onnxDataTypes) {
                "Wrong $what ONNX data type '${value.name}' for '${info.type}' operator\nPresent: ${value.type}, Expected: [${constraint.onnxDataTypes.joinToString()}]"
            }
        }
    }

    suspend fun <D : ONNXData<*, *>> applyWithCheck(contexts: Contexts<D>, inputs: List<T?>): List<U?> {
        check(info.inputs, inputs, "input")
        val outputs = apply(contexts, inputs)
        require(outputs.size >= this.outputs.size) {
            "Operator '${info.type}' doesn't provide expected output size\nPresent: ${outputs.size}, Expected: at least ${this.outputs.size}"
        }
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
        requireNotNull(info) { "Attribute '$key' not specified in the '${this.info.type}' operator" }

        return attributes[key]?.value as T? ?: if (!info.required) info.default as T? else null
    }

    fun hasAttributeSet(key: String): Boolean {
        val info = info.attributes[key]
        requireNotNull(info) { "Attribute '$key' not specified in the '${this.info.type}' operator" }

        return attributes[key]?.value != null
    }

    abstract suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<T?>): List<U?>
    open suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, vararg inputs: T?): Collection<U?> = apply(contexts, inputs.toList())

    override suspend fun close() {
        for (attribute in attributes.values) {
            attribute.close()
        }

        for (attributeInfo in info.attributes.values) {
            attributeInfo.close()
        }
    }

    companion object {
        private val logger = LoggerFactory.create("io.kinference.operator.Operator")

        val ALL_DATA_TYPES = TensorProto.DataType.values().toHashSet() - TensorProto.DataType.UNDEFINED
        val PRIMITIVE_DATA_TYPES = setOf(
            TensorProto.DataType.BOOL,
            TensorProto.DataType.BFLOAT16,
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
        val UINT_DATA_TYPES = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.UINT16,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
        )

        val INT_DATA_TYPES = setOf(
            TensorProto.DataType.INT8,
            TensorProto.DataType.INT16,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
        )

        val NUMBER_DATA_TYPES = INT_DATA_TYPES + UINT_DATA_TYPES + FLOAT_DATA_TYPES
    }
}
