import io.gitlab.arturbosch.detekt.Detekt
import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    kotlin("jvm") version "1.7.10"
    kotlin("plugin.serialization") version "1.7.10"
    java
    id("com.github.johnrengelman.shadow") version "7.1.2"
    jacoco
    id("org.jetbrains.dokka") version "1.7.10"
    id("io.gitlab.arturbosch.detekt") version "1.21.0"
    id("com.github.sherter.google-java-format") version "0.9"
//    kotlin("jupyter.api") version "0.10.1-8"
    id("com.github.jk1.dependency-license-report") version "2.1"
    id("com.github.spotbugs") version "5.0.9"
    application
}

group = "jp.live.ugai"
version = "1.0-SNAPSHOT"
val v = "0.19.0-SNAPSHOT"

val ktlint by configurations.creating

repositories {
    mavenCentral()
    maven {
        url = uri("https://oss.sonatype.org/content/repositories/snapshots/")
    }
}

dependencies {
    implementation("ai.djl:basicdataset:$v")
    implementation("ai.djl:api:$v")
//    implementation("ai.djl.mxnet:mxnet-engine:$v")
    implementation("ai.djl.pytorch:pytorch-engine:$v")
    runtimeOnly("ai.djl.pytorch:pytorch-jni:1.12.1-$v")
//    runtimeOnly("ai.djl.pytorch:pytorch-native-cpu:1.12.1")
    //    implementation("ai.djl.pytorch:pytorch-native-cpu:1.12.1:linux-x86_64")
//    runtimeOnly("ai.djl.pytorch:pytorch-native-cu116:1.12.1:linux-x86_64")
    implementation("org.slf4j:slf4j-simple:1.7.36")
    implementation(kotlin("stdlib-jdk8"))
    implementation("com.opencsv:opencsv:5.6")
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.9.0")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.9.0")
    ktlint("com.pinterest:ktlint:0.47.0") {
        attributes {
            attribute(Bundling.BUNDLING_ATTRIBUTE, objects.named(Bundling.EXTERNAL))
        }
    }
}

tasks {
    compileKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }

    compileTestKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }

    compileJava {
        options.encoding = "UTF-8"
        sourceCompatibility = "1.8"
        targetCompatibility = "1.8"
    }

    compileTestJava {
        options.encoding = "UTF-8"
        sourceCompatibility = "1.8"
        targetCompatibility = "1.8"
    }

    test {
        useJUnitPlatform()
        finalizedBy(jacocoTestReport) // report is always generated after tests run
    }

    withType<Detekt>().configureEach {
        // Target version of the generated JVM bytecode. It is used for type resolution.
        jvmTarget = "1.8"
        reports {
            // observe findings in your browser with structure and code snippets
            html.required.set(true)
            // checkstyle like format mainly for integrations like Jenkins
            xml.required.set(true)
            // similar to the console output, contains issue signature to manually edit baseline files
            txt.required.set(true)
            // standardized SARIF format (https://sarifweb.azurewebsites.net/) to support integrations
            // with Github Code Scanning
            sarif.required.set(true)
        }
    }

    check {
        dependsOn("ktlint")
    }

    jacocoTestReport {
        dependsOn(test) // tests are required to run before generating the report
    }

    withType<ShadowJar> {
        manifest {
            attributes["Main-Class"] = "com.fujitsu.labs.virtualhome.MainKt"
        }
    }
}

task("ktlint", JavaExec::class) {
    group = "verification"
    description = "Check Kotlin code style."
    classpath = ktlint
    mainClass.set("com.pinterest.ktlint.Main")
    args = listOf("src/**/*.kt")
}

val ktlintFormat by tasks.creating(JavaExec::class) {
    description = "Fix Kotlin code style deviations."
    classpath = ktlint
    mainClass.set("com.pinterest.ktlint.Main")
    args = listOf("-F", "src/**/*.kt")
}

detekt {
    buildUponDefaultConfig = true // preconfigure defaults
    allRules = false // activate all available (even unstable) rules.
    // point to your custom config defining rules to run, overwriting default behavior
    config = files("$projectDir/config/detekt.yml")
//    baseline = file("$projectDir/config/baseline.xml") // a way of suppressing issues before introducing detekt
}

spotbugs {
    ignoreFailures.set(true)
}

jacoco {
    toolVersion = "0.8.8"
//    reportsDirectory.set(layout.buildDirectory.dir("customJacocoReportDir"))
}

application {
    mainClass.set("jp.live.ugai.tugraph.Test6Kt")
}
