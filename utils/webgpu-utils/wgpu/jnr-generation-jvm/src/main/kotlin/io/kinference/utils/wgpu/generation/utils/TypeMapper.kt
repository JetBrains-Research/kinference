package io.kinference.utils.wgpu.generation.utils

import io.kinference.utils.wgpu.generation.models.*

class TypeMapper(declarations: LibraryDeclarations) {
    private val enums = declarations.enumDeclarations.map { it.name }.toSet()
    private val structs = declarations.structDeclarations.map { it.name }.toSet()
    private val typeAliases = declarations.typeAliases

    fun getType(fieldType: TypeRepresentation): Type {
        val typeName = fieldType.name
        return when {
            typeName == "WGPUChainedStruct" && fieldType.pointer -> PointerType
            typeName == "WGPUChainedStructOut" && fieldType.pointer -> PointerType
            structs.contains(typeName) -> {
                if (fieldType.pointer) {
                    StructPointerType(typeName)
                } else {
                    StructType(typeName)
                }
            }
            typeAliases.containsKey(typeName) -> {
                val typeAlias = typeAliases.getValue(typeName)
                getType(TypeRepresentation(typeAlias.name, fieldType.pointer || typeAlias.pointer))
            }
            fieldType.pointer -> {
                if (typeName == "char") {
                    CStringType
                } else {
                    PointerType
                }
            }
            enums.contains(typeName) -> EnumType(typeName)
            typeName == "bool" -> BooleanType
            typeName == "uint16_t" -> Unsigned16Type
            typeName == "uint32_t" -> Unsigned32Type
            typeName == "uint64_t" -> Unsigned64Type
            typeName == "int16_t" -> Signed16Type
            typeName == "int32_t" -> Signed32Type
            typeName == "int64_t" -> Signed64Type
            typeName == "float" -> FloatType
            typeName == "double" -> DoubleType
            else -> error("Unrecognized type: $typeName")
        }
    }
}
