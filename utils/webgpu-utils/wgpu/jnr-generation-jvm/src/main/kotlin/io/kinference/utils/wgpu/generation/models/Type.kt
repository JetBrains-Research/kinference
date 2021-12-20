package io.kinference.utils.wgpu.generation.models

sealed class Type

data class StructType(val name: String) : Type()
data class StructPointerType(val name: String) : Type()

data class EnumType(val name: String) : Type()

object PointerType : Type()
object CStringType : Type()

object BooleanType : Type()

object Unsigned16Type : Type()
object Unsigned32Type : Type()
object Unsigned64Type : Type()

object Signed16Type : Type()
object Signed32Type : Type()
object Signed64Type : Type()

object FloatType : Type()
object DoubleType : Type()
