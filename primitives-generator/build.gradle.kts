plugins {
    `maven-publish`
    `java-gradle-plugin`
    kotlin("jvm") apply true
}

group = "org.jetbrains.research.kotlin.inference"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(project(":annotations"))
    implementation("com.squareup:kotlinpoet:1.6.0")

    api(kotlin("compiler-embeddable"))
    api(kotlin("gradle-plugin-api"))
}

gradlePlugin {
    plugins {
        create("primitives-plugin") {
            id = "org.jetbrains.research.kotlin.inference.primitives-generator"
            implementationClass = "org.jetbrains.research.kotlin.inference.PrimitivesKotlinGradlePlugin"
        }
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            groupId = "org.jetbrains.research.kotlin.inference"
            artifactId = "primitives-generator"
            version = "0.1.0"

            from(components["kotlin"])
        }
    }

    repositories {
        mavenLocal()
    }
}
