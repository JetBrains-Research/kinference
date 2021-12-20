package io.kinference.utils.wgpu.generation.models

data class EnumDeclaration(
    val name: String,
    val options: List<EnumOption>,
)

data class EnumOption(
    val name: String,
    val value: String,
)
