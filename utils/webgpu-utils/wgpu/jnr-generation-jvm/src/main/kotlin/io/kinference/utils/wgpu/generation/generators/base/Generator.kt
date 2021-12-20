package io.kinference.utils.wgpu.generation.generators.base

abstract class Generator<T> {
    protected abstract fun generateImpl()
    abstract fun generate(): T
}
