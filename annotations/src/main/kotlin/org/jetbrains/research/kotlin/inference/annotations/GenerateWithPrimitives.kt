package org.jetbrains.research.kotlin.inference.annotations

@Target(AnnotationTarget.FILE)
@Retention(AnnotationRetention.SOURCE)
annotation class GenerateWithPrimitives

@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.SOURCE)
annotation class PrimitiveClass
