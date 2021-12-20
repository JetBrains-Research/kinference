package io.kinference.utils.wgpu.generation.generators.base

import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.TypeSpec

abstract class ClassGenerator(private val className: ClassName) : TypeGenerator() {
    override fun initBuilder() {
        builder = TypeSpec.classBuilder(className)
    }
}
