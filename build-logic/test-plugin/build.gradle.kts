plugins {
    `kotlin-dsl`
    `java-gradle-plugin`
}

dependencies {
    // Use the Kotlin Gradle plugin from the version catalog
    implementation(libs.kotlin.gradle.plugin)
    
    // AWS S3 dependency for test data
    implementation(libs.aws.s3)
    
    // Gradle API and Kotlin DSL
    implementation(gradleApi())
    implementation(gradleKotlinDsl())
}

gradlePlugin {
    plugins {
        create("testPlugin") {
            id = "io.kinference.testplugin"
            implementationClass = "io.kinference.gradle.TestPlugin"
        }
    }
}
