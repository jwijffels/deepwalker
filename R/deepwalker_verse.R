#' @title Versatile Graph Embeddings from Similarity Measures
#' @description This function calculates node embeddings in a graph by using 
#' VERSE (Versatile Graph Embeddings from Similarity Measures)
#' @param x an object of class dgRMatrix
#' @param dimension integer with the dimension of the embedding
#' @param epochs number of epochs
#' @param learning_rate the learning rate of the algorithm
#' @param samples integer
#' @param alpha the alpha parameter of the algorithm
#' @return 
#' a list with elements 
#' \enumerate{
#' \item{vertices: }{the number of vertices in the graph}
#' \item{edges: }{the number of edges (links) between the vertices in the graph}
#' \item{embedding: }{the embeddings of the vertices}
#' \item{edges_data and offsets_data: }{for debugging purposes now}
#' }
#' @export
#' @examples
#' library(igraphdata)
#' library(igraph)
#' library(Matrix)
#' 
#' data(karate, package = "igraphdata")
#' x <- as_adj(karate)
#' x <- as(x, "RsparseMatrix")
#' 
#' embeddings <- deepwalker_verse(x, dimension = 64, epochs = 1000)
deepwalker_verse <- function(x, 
                             similarity = c("pagerank", "adjacency", "simrank"),
                             dimension = 128,
                             epochs = 100000, 
                             learning_rate = 0.0025,
                             samples = 3,
                             alpha = 0.85){
  stopifnot(inherits(x, "dgRMatrix"))
  similarity <- match.arg(similarity)
  if(similarity == "pagerank"){
    emb <- embeddings_verse_pagerank(x@p, x@j, 
                            dimension = dimension, 
                            epochs = epochs,
                            learning_rate = learning_rate,
                            samples = samples,
                            alpha = alpha)
  }else if(similarity == "adjacency"){
    
  }else if(similarity == "simrank"){
    
  }
  
  emb
}