#' @title Versatile Graph Embeddings from Similarity Measures
#' @description This function calculates node embeddings in a graph by using 
#' VERSE (Versatile Graph Embeddings from Similarity Measures)
#' @param x an object of class dgRMatrix
#' @return TODO
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
#' embeddings <- deepwalker_verse(x)
deepwalker_verse <- function(x){
  emb <- embeddings_verse(x@p, x@j, n_epochs = 10)
  emb
}