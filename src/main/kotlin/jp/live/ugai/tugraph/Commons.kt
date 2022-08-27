package jp.live.ugai.tugraph

// Hyper Parameters
const val LEARNING_RATE = 0.001f
const val DIMENSION = 100L
const val NEPOCH = 1000
const val BATCH_SIZE = 3
const val TRIPLE = 3L
const val NUM_ENTITIES = 4L

//const val NUM_ENTITIES = 7658L
const val NUM_EDGES = 2L

//const val NUM_EDGES = 57L
const val NUM_TRIPLE = 5L
val triples = longArrayOf(2, 0, 1, 2, 1, 3, 0, 0, 1, 0, 1, 2, 1, 0, 2)
val labelData = floatArrayOf(0f, 0f, 1f, 0f, 0f)

class Commons
