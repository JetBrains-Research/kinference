package io.kinference.types

import io.kinference.graph.GraphContext
import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.protobuf.message.TensorShapeProto
import io.kinference.protobuf.message.TypeProto

class TensorShape internal constructor(private val dims: List<Dimension>? = null) {
    constructor(shape: IntArray) : this(shape.map { StaticDimension(it) })

    val size: Int
        get() = dims?.size ?: -1

    open class Dimension
    class StaticDimension(val value: Int) : Dimension()
    class DynamicDimension(val value: String) : Dimension()
    object UnknownDimension : Dimension()

    fun getDimensions(context: GraphContext<*>? = null): IntArray {
        if (context == null) require(dims!!.all { it is StaticDimension })
        return dims!!.map {
            when (it) {
                is StaticDimension -> it.value
                is DynamicDimension -> context!!.getShape(it.value)
                is UnknownDimension -> -1
                else -> error("Unsupported dimension type")
            }
        }.toIntArray()
    }

    companion object {
        fun empty() = TensorShape(emptyList())
        fun unknown() = TensorShape(null)

        operator fun invoke(proto: TensorShapeProto): TensorShape {
            return TensorShape(proto.dim.map {
                when {
                    it.dimValue != null -> StaticDimension(it.dimValue!!.toInt())
                    it.dimParam != null -> DynamicDimension(it.dimParam!!)
                    else -> UnknownDimension
                }
            })
        }
    }
}

sealed class ValueTypeInfo {
    companion object {
        fun create(proto: TypeProto) = when {
            proto.tensorType != null -> TensorTypeInfo(proto.tensorType!!)
            proto.sequenceType != null -> SequenceTypeInfo(proto.sequenceType!!)
            proto.mapType != null -> MapTypeInfo(proto.mapType!!)
            else -> error("One should be present")
        }
    }

    class TensorTypeInfo(shape: TensorShape? = null, type: DataType? = null) : ValueTypeInfo() {
        constructor(proto: TypeProto.Tensor) : this(proto.shape?.let { TensorShape(it) } ?: TensorShape.empty(), proto.elem_type!!)

        val type = type ?: DataType.UNDEFINED
        val shape = shape ?: TensorShape.empty()
    }

    class SequenceTypeInfo(elementType: ValueTypeInfo? = null) : ValueTypeInfo() {
        constructor(proto: TypeProto.Sequence) : this(create(proto.elem_type!!))

        val elementType = elementType ?: DataType.UNDEFINED
    }

    class MapTypeInfo(keyType: DataType? = null, val valueType: ValueTypeInfo? = null) : ValueTypeInfo() {
        constructor(proto: TypeProto.Map) : this(DataType.fromValue(proto.key_type!!)!!, create(proto.value_type!!))

        val keyType = keyType ?: DataType.UNDEFINED
    }
}
