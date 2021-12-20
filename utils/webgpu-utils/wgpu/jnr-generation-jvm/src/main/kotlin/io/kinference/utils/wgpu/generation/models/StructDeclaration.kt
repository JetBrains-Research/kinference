package io.kinference.utils.wgpu.generation.models

data class StructDeclaration(
    val name: String,
    val fields: List<StructField>
)

data class StructField(
    val name: String,
    val type: TypeRepresentation,
)
