package io.kinference.utils.wgpu.generation.generators.base

import com.squareup.kotlinpoet.TypeSpec
import io.kinference.utils.wgpu.generation.generators.base.Generator

abstract class TypeGenerator : Generator<TypeSpec>() {
    protected lateinit var builder: TypeSpec.Builder

    abstract fun initBuilder()

    override fun generate(): TypeSpec {
        initBuilder()
        generateImpl()
        return builder.build()
    }
}
