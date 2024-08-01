package io.kinference.gradle

import org.gradle.api.NamedDomainObjectContainer
import org.gradle.api.NamedDomainObjectProvider
import org.jetbrains.kotlin.gradle.plugin.KotlinSourceSet

val NamedDomainObjectContainer<KotlinSourceSet>.commonMain: NamedDomainObjectProvider<KotlinSourceSet>
    get() = this.named("commonMain")

val NamedDomainObjectContainer<KotlinSourceSet>.commonTest: NamedDomainObjectProvider<KotlinSourceSet>
    get() = this.named("commonTest")

val NamedDomainObjectContainer<KotlinSourceSet>.jvmMain: NamedDomainObjectProvider<KotlinSourceSet>
    get() = this.named("jvmMain")

val NamedDomainObjectContainer<KotlinSourceSet>.jvmTest: NamedDomainObjectProvider<KotlinSourceSet>
    get() = this.named("jvmTest")

val NamedDomainObjectContainer<KotlinSourceSet>.jsMain: NamedDomainObjectProvider<KotlinSourceSet>
    get() = this.named("jsMain")

val NamedDomainObjectContainer<KotlinSourceSet>.jsTest: NamedDomainObjectProvider<KotlinSourceSet>
    get() = this.named("jsTest")
