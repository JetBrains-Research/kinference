package io.kinference.utils.wgpu.generation.generators.base

import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.TypeSpec

abstract class EnumClassGenerator(private val className: ClassName) : TypeGenerator() {
    override fun initBuilder() {
        builder = TypeSpec.enumBuilder(className)
    }
}
