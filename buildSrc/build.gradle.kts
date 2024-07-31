repositories {
    mavenCentral()
}

plugins {
    `kotlin-dsl`
}

dependencies {
    implementation(libs.kotlin.gradle.plugin)
    implementation(gradleApi())
    implementation(gradleKotlinDsl())
    api(libs.aws.s3)
}
