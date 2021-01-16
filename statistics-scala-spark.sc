import org.apache.spark.{SparkConf, SparkContext}

val conf = new SparkConf()
conf.setMaster("local")
conf.setAppName("first-spark")

val sc = new SparkContext(conf)

// STATISTICS-------------------------------------
// imports:
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.linalg.{Matrix,Matrices}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
// RDD vector:
val observacoes: RDD[Vector] = sc.parallelize(Array(
  Vectors.dense(1.0,2.0),
  Vectors.dense(5.0,6.0),
  Vectors.dense(9.0,10.0)))
// SUMMARY STATISTICS EXAMPLE::::::
val summary: MultivariateStatisticalSummary = Statistics.colStats(observacoes)
summary.mean
summary.variance
summary.numNonzeros
summary.normL1
summary.normL2

// CORRELATION ------------------------------------------------------
// PEARSON CORRELATION BETWEEN TWO SERIES
val X: RDD[Double] = sc.parallelize(Array(2.0,9.0,-7.0))
val Y: RDD[Double] = sc.parallelize(Array(1.0,3.0,5.0))
val correlation: Double = Statistics.corr(X,Y,"pearson")
// PEARSON CORRELATION AMONG SERIES
val data: RDD[Vector] = sc.parallelize(Array(
  Vectors.dense(1.0,-2.0,3.0),
  Vectors.dense(6.0,8.0,-1.0),
  Vectors.dense(4.0,0.0,-5.0)))
val correlMatrix: Matrix = Statistics.corr(data,"pearson")
// SPEARMAN CORRELATION (Ideal para correlacao entre mais de 2 series)
val ranks: RDD[Vector] = sc.parallelize(Array(
  Vectors.dense(1.0,2.0,3.0),
  Vectors.dense(4.0,5.0,6.0),
  Vectors.dense(7.0,8.0,9.0)))
val correlSpearman: Matrix = Statistics.corr(ranks,"spearman")


// GERANDO DADOS ALEATÓRIS -----------------------------------------
// SIMPLE EXEMPLE
import org.apache.spark.mllib.random.RandomRDDs._
val million = poissonRDD(sc, mean=1.0, size =  1000000L,
                           numPartitions = 10)
// STATISTICS
million.mean()
million.variance()
// SIMPLE VECTOR EXAMPLE
val dados = normalVectorRDD(sc, numRows = 10000, numCols = 3,
                              numPartitions = 10)
val stats: MultivariateStatisticalSummary = Statistics.colStats(dados)
stats.mean
stats.variance
// EXISTEM MUUUUUUUUITOS TIPOS DE VETORES RDD (GAMMA, POISSON, LOGNORMAL, ETC)


// SAMPLING / SPLITTING ----------------------------------------------
val arrayAleatorio = sc.parallelize(1 to 1000000)
// dividir o valor de "arrayAleatorio" em 3 arrays, contendo 60%, 20% e 20% do total
val splits = arrayAleatorio.randomSplit(Array(0.6,0.2,0.2), seed = 13L)
// Criando as variaveis para receber os valores:
val training = splits(0)
val test = splits(1)
val validation = splits(2)
// Exibir as contagens dos splits:
splits.map(_.count())
// STRATIFIED SAMPLING ------------------------------------------------
// podem ser performados em Key-value pairs do RDD, como no exemplo:
import org.apache.spark.mllib.linalg.distributed.IndexedRow
// criando RDD indexedrow (duas com o mesmo index "1")
val rowss: RDD[IndexedRow] = sc.parallelize(Array(
  IndexedRow(0, Vectors.dense(1.0,2.0)),
  IndexedRow(1, Vectors.dense(4.0,5.0)),
  IndexedRow(1, Vectors.dense(7.0,8.0))))
// probabilidade de ter o elemento da chave associada (100% para 0, 50% para 1(pq tem 2 index 1)
val fractions: Map[Long, Double] = Map(0L -> 1.0, 1L -> 5.0)
// mapeando em chave-valor padrao antes de fazer a amostragem:
val aproxxSample = rowss.map{
  case IndexedRow(index, vec) => (index, vec)
}.sampleByKey(withReplacement = false, fractions, 9L)
// resultado da amostragem:
aproxxSample.collect()


// HYPOTHESIS TESTING-------------------------------------------------
// Person's Chi-Squared test for goodness of fit:
// Determina se uma distribuição de frequência observada difere de uma determinada distribuição ou não
// INPUT É UM VECTOR
val vect: Vector = Vectors.dense(
  0.3, 0.2, 0.15, 0.01, 0.01, 0.01, 0.05)
val goodnessFitTest =  Statistics.chiSqTest(vect)

// Person's Chi-Squared test for independence
// Determina se observações desemparelhadas em duas variáveis são independentes uma da outra
// INPUT É UMA MATRIX
val mat: Matrix = Matrices.dense(3,2,
  Array(13.0, 47.0, 40.0, 80.0, 11.0, 9.0))
val independenceTestResult =  Statistics.chiSqTest(mat)

// Kolmogorov-Smirnov test for equality distribution
// Determina se duas distribuições de probabilidade são iguais ou não
val datas: RDD[Double] = normalRDD(sc, size=100,
  numPartitions = 1, seed = 13L)
val testResult = Statistics.kolmogorovSmirnovTest(datas, "norm", 0, 1)

// Kernel Density Estimation
// Calcular uma estimativa da função densidade de probabilidade de uma variável aleatória...
// ... avaliadade em um determinado conjunto de pontos
// No Spark, somente o kernel de GAUSS é suportado
import org.apache.spark.mllib.stat.KernelDensity
// DoubleRDD standard normal distribution
val data2: RDD[Double] = normalRDD(sc, size=1000,
  numPartitions = 1, seed = 17)
// Nova instancia do kernel densitu, passando tambem os dados com um Sample e...
//... o desvio padrao do kernel Gaussiano (setBandwidth(0.1))
val kd = new KernelDensity().setSample(data2).setBandwidth(0.1)
// usando 7 pontos de validação, para estimar a função densidade de probabilidade
val densities = kd.estimate(
Array(-1.5, -1, -0.5, 0, 0.5, 1, 1.5))

