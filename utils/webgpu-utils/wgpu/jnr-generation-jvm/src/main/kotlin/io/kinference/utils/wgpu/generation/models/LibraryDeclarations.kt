package io.kinference.utils.wgpu.generation.models

data class LibraryDeclarations(
    val enumDeclarations: List<EnumDeclaration>,
    val structDeclarations: List<StructDeclaration>,
    val typeAliases: Map<String, TypeRepresentation>,
)
